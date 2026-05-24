//! Multi-model HTTP server: one GPU, several models resident, requests
//! serialised through a single FIFO queue.
//!
//! Three ports — Big LLM, Small LLM, Embedder — each its own listener.
//! Every request is pushed onto one shared channel; a single worker
//! thread owns the GPU, pulls jobs in order, runs the target model, and
//! sends the response back to the waiting connection. Models never run
//! concurrently (one GPU), so the worker simply blocks per job.
//!
//! API is OpenAI-shaped: `POST /v1/completions` on the LLM ports,
//! `POST /v1/embeddings` on the embedder port. The embedder is a
//! follow-up (nomic-bert is a new encoder architecture) — its port
//! answers 503 until then.

mod http;
mod json;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use json::Json;

use crate::gguf::GgufFile;
use crate::runtime::KernelCache;

/// Server-wide counters, shared between worker + acceptors. Atomic
/// so the `/metrics` endpoint can read them without locking.
#[derive(Default)]
struct Metrics {
    requests_total:    AtomicU64,    // every HTTP request reaching a route
    requests_ok:       AtomicU64,    // 2xx replies
    requests_4xx:      AtomicU64,
    requests_5xx:      AtomicU64,
    prompt_tokens:     AtomicU64,    // total across all completed requests
    completion_tokens: AtomicU64,
    decode_us_total:   AtomicU64,    // sum of decode wall times (microseconds)
    requests_eos:      AtomicU64,    // finish_reason == stop
    requests_length:   AtomicU64,    // finish_reason == length
    panics_recovered:  AtomicU64,    // catch_unwind hits in the worker
    start_unix:        AtomicU64,    // server up-time anchor
}

impl Metrics {
    fn new() -> Self {
        let m = Self::default();
        m.start_unix.store(unix_now(), Ordering::Relaxed);
        m
    }

    /// Prometheus-style text exposition. Cheap — read each counter once.
    fn render_prometheus(&self) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(2048);
        let metric = |s: &mut String, name: &str, help: &str, value: u64| {
            let _ = writeln!(s, "# HELP reinstinct_{name} {help}");
            let _ = writeln!(s, "# TYPE reinstinct_{name} counter");
            let _ = writeln!(s, "reinstinct_{name} {value}");
        };
        metric(&mut s, "requests_total", "total HTTP requests reaching a route",
               self.requests_total.load(Ordering::Relaxed));
        metric(&mut s, "requests_ok_total", "requests with a 2xx reply",
               self.requests_ok.load(Ordering::Relaxed));
        metric(&mut s, "requests_4xx_total", "requests with a 4xx reply",
               self.requests_4xx.load(Ordering::Relaxed));
        metric(&mut s, "requests_5xx_total", "requests with a 5xx reply",
               self.requests_5xx.load(Ordering::Relaxed));
        metric(&mut s, "prompt_tokens_total", "sum of prompt_tokens across completed requests",
               self.prompt_tokens.load(Ordering::Relaxed));
        metric(&mut s, "completion_tokens_total", "sum of completion_tokens",
               self.completion_tokens.load(Ordering::Relaxed));
        metric(&mut s, "decode_us_total", "sum of decode wall time (microseconds)",
               self.decode_us_total.load(Ordering::Relaxed));
        metric(&mut s, "requests_eos_total", "requests that ended at EOS",
               self.requests_eos.load(Ordering::Relaxed));
        metric(&mut s, "requests_length_total", "requests stopped by max_tokens or timeout",
               self.requests_length.load(Ordering::Relaxed));
        metric(&mut s, "panics_recovered_total", "panics caught by the worker's catch_unwind",
               self.panics_recovered.load(Ordering::Relaxed));
        let _ = writeln!(s, "# HELP reinstinct_start_unix_seconds server start time");
        let _ = writeln!(s, "# TYPE reinstinct_start_unix_seconds gauge");
        let _ = writeln!(s, "reinstinct_start_unix_seconds {}",
                         self.start_unix.load(Ordering::Relaxed));
        s
    }
}

/// Which resident model a request targets.
#[derive(Clone, Copy, PartialEq)]
enum Target { Big, Small, Embed }

impl Target {
    fn label(self) -> &'static str {
        match self { Target::Big => "big", Target::Small => "small", Target::Embed => "embed" }
    }
}

/// What the client sent as the prompt — either a raw text completion
/// (`/v1/completions` POST `prompt`) or a chat-completions message
/// array (`/v1/chat/completions` POST `messages`). The worker uses
/// this variant to pick the response shape (`text_completion` vs
/// `chat.completion`) AND, for the chat path, to apply the right
/// per-architecture chat template before tokenization.
enum PromptInput {
    Raw(String),
    Chat(Vec<crate::chat::ChatMessage>),
}

/// A parsed `/v1/completions` or `/v1/chat/completions` request.
struct GenReq {
    prompt: PromptInput,
    max_tokens: usize,
    sampler: crate::sampling::SamplerParams,
    /// MTP spec-decode opt-in/opt-out. `None` ⇒ use the server default
    /// (true if the target has a drafter loaded, false otherwise).
    /// `Some(false)` lets a per-turn classifier disable the drafter on
    /// creative work where it would just waste verify cycles.
    use_speculative: Option<bool>,
    /// Per-request K override for spec-decode. `None` ⇒ server default
    /// (currently 3). Ignored when spec-decode is off.
    speculative_k: Option<usize>,
    /// Wall-clock cap on the generation. If decode runs past it, the
    /// request stops early with `finish_reason: "length"`. None disables.
    request_timeout: Option<std::time::Duration>,
    /// OpenAI `stream: true` — Server-Sent Events response. The worker
    /// emits one `Chunk` per decoded token; the connection writes them
    /// as `data: {json}\n\n`. Implicit cancellation: if the client closes
    /// the socket, the next Chunk send fails (Receiver dropped) and the
    /// worker stops generating.
    stream: bool,
}

/// Messages from the GPU worker to the connection-handler thread.
/// Non-streaming requests send exactly one `Done(reply)`. Streaming
/// requests send a sequence of `Chunk` (one per decoded token) then a
/// final `Done` carrying the trailing usage stats.
enum StreamMsg {
    /// SSE-style chunk — the connection writes `data: {payload}\n\n`.
    Chunk(String),
    /// Final reply for non-streaming, or stream-terminator for streaming.
    Done(HttpReply),
}

impl GenReq {
    fn is_chat(&self) -> bool { matches!(self.prompt, PromptInput::Chat(_)) }
}

/// A unit of work handed from a connection thread to the GPU worker.
struct Job {
    /// Monotonic id for log + metric correlation.
    request_id: u64,
    target: Target,
    /// `Err` carries an already-formed client error (bad request / wrong route).
    req: Result<GenReq, (u16, &'static str, String)>,
    reply: mpsc::Sender<StreamMsg>,
}

struct HttpReply {
    status: u16,
    status_text: &'static str,
    body: String,
}

// --- request parsing ---------------------------------------------------

/// Server-wide cap on a single request's wall-clock generation. Bigger
/// than this and the worker can't service incoming requests; clients
/// can ask for less via `request_timeout_seconds`.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Common decode + sampling fields shared by both endpoint parsers.
/// Returns the parsed (GenReq fields). Accepts every OpenAI sampler
/// knob plus a few extensions (min_p, repetition_penalty, mirostat).
fn parse_common_fields(j: &Json)
    -> (usize, crate::sampling::SamplerParams, Option<bool>, Option<usize>,
        Option<std::time::Duration>, bool)
{
    use crate::sampling::{SamplerParams, MirostatV2};
    let max_tokens = j.get("max_tokens").and_then(Json::as_f64)
        .map(|n| n as usize).unwrap_or(256).clamp(1, 4096);

    let mut sp = SamplerParams::default();
    sp.temperature = j.get("temperature").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(0.8).max(0.0);
    sp.top_k = j.get("top_k").and_then(Json::as_f64)
        .map(|n| n as usize).unwrap_or(40);
    sp.top_p = j.get("top_p").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(1.0).clamp(0.0, 1.0);
    sp.min_p = j.get("min_p").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(0.0).clamp(0.0, 1.0);
    sp.repetition_penalty = j.get("repetition_penalty").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(1.0).max(0.0);
    sp.repetition_window = j.get("repetition_window").and_then(Json::as_f64)
        .map(|n| n as usize).unwrap_or(64);
    sp.frequency_penalty = j.get("frequency_penalty").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(0.0);
    sp.presence_penalty = j.get("presence_penalty").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(0.0);
    sp.seed = j.get("seed").and_then(Json::as_f64).map(|n| n as u64).unwrap_or(0);

    // Mirostat v2: opt-in via `mirostat: 2`. tau + eta override defaults.
    if j.get("mirostat").and_then(Json::as_f64).map(|n| n as i64) == Some(2) {
        let tau = j.get("mirostat_tau").and_then(Json::as_f64)
            .map(|n| n as f32).unwrap_or(5.0);
        let eta = j.get("mirostat_eta").and_then(Json::as_f64)
            .map(|n| n as f32).unwrap_or(0.1);
        sp.mirostat = Some(MirostatV2::new(tau, eta));
    }

    let use_speculative = j.get("use_speculative").and_then(Json::as_bool);
    let speculative_k = j.get("speculative_k").and_then(Json::as_f64)
        .map(|n| (n as usize).clamp(1, 4));
    let request_timeout = j.get("request_timeout_seconds").and_then(Json::as_f64)
        .map(|n| std::time::Duration::from_secs_f64(n.max(0.1).min(600.0)))
        .or(Some(std::time::Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)));
    let stream = j.get("stream").and_then(Json::as_bool).unwrap_or(false);
    (max_tokens, sp, use_speculative, speculative_k, request_timeout, stream)
}

// --- streaming helpers (SSE) --------------------------------------------

/// One streamed text-completion chunk in OpenAI shape. Each event is a
/// separate `data: ...` line; the client SDK concatenates `text` fields.
fn completion_stream_chunk(id: &str, model: &str, text: &str, finish: Option<&str>) -> String {
    let mut choice = vec![
        ("text".into(),     Json::Str(text.to_string())),
        ("index".into(),    Json::Num(0.0)),
        ("logprobs".into(), Json::Null),
    ];
    choice.push(("finish_reason".into(),
                 finish.map(|f| Json::Str(f.to_string())).unwrap_or(Json::Null)));
    Json::Obj(vec![
        ("id".into(),      Json::Str(id.to_string())),
        ("object".into(),  Json::Str("text_completion".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![Json::Obj(choice)])),
    ]).to_string()
}

/// One streamed chat-completion chunk. First chunk carries `role`;
/// subsequent chunks carry just `content`; final chunk has `finish_reason`.
fn chat_stream_chunk(id: &str, model: &str, delta: ChatDelta, finish: Option<&str>) -> String {
    let mut d = Vec::with_capacity(2);
    if let Some(r) = delta.role    { d.push(("role".into(),    Json::Str(r.to_string()))); }
    if let Some(c) = delta.content { d.push(("content".into(), Json::Str(c.to_string()))); }
    let mut choice = vec![
        ("index".into(), Json::Num(0.0)),
        ("delta".into(), Json::Obj(d)),
    ];
    choice.push(("finish_reason".into(),
                 finish.map(|f| Json::Str(f.to_string())).unwrap_or(Json::Null)));
    Json::Obj(vec![
        ("id".into(),      Json::Str(id.to_string())),
        ("object".into(),  Json::Str("chat.completion.chunk".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![Json::Obj(choice)])),
    ]).to_string()
}

struct ChatDelta<'a> { role: Option<&'a str>, content: Option<&'a str> }

/// Parse an OpenAI `/v1/completions` body into a `GenReq`. Raw-prompt
/// path; no chat template is applied server-side.
fn parse_completions(body: &str) -> Result<GenReq, (u16, &'static str, String)> {
    let bad = |m: String| (400u16, "Bad Request", m);
    let j = Json::parse(body).map_err(|e| bad(format!("invalid JSON: {e}")))?;
    let prompt = j.get("prompt").and_then(Json::as_str)
        .ok_or_else(|| bad("missing string field 'prompt'".into()))?
        .to_string();
    let (max_tokens, sampler, use_speculative, speculative_k, request_timeout, stream)
        = parse_common_fields(&j);
    Ok(GenReq { prompt: PromptInput::Raw(prompt), max_tokens, sampler,
                use_speculative, speculative_k, request_timeout, stream })
}

/// Parse an OpenAI `/v1/chat/completions` body into a `GenReq`. The
/// `messages` array becomes a `PromptInput::Chat`; the worker's
/// model knows which per-architecture chat template to apply.
fn parse_chat_completions(body: &str) -> Result<GenReq, (u16, &'static str, String)> {
    use crate::chat::{ChatMessage, Role};
    let bad = |m: String| (400u16, "Bad Request", m);
    let j = Json::parse(body).map_err(|e| bad(format!("invalid JSON: {e}")))?;
    let messages_arr = j.get("messages")
        .ok_or_else(|| bad("missing array field 'messages'".into()))?;
    let arr = match messages_arr {
        Json::Arr(a) => a,
        _ => return Err(bad("'messages' must be an array".into())),
    };
    if arr.is_empty() {
        return Err(bad("'messages' must contain at least one message".into()));
    }
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let role_s = m.get("role").and_then(Json::as_str)
            .ok_or_else(|| bad(format!("messages[{i}]: missing string 'role'")))?;
        let content = m.get("content").and_then(Json::as_str)
            .ok_or_else(|| bad(format!("messages[{i}]: missing string 'content'")))?;
        let role = match role_s {
            "system"    => Role::System,
            "user"      => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(bad(format!(
                "messages[{i}]: unknown role '{other}' (want system|user|assistant)"))),
        };
        messages.push(ChatMessage { role, content: content.to_string() });
    }
    let (max_tokens, sampler, use_speculative, speculative_k, request_timeout, stream)
        = parse_common_fields(&j);
    Ok(GenReq { prompt: PromptInput::Chat(messages), max_tokens, sampler,
                use_speculative, speculative_k, request_timeout, stream })
}

// --- OpenAI response shaping -------------------------------------------

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn completion_response(model: &str, text: &str, n_prompt: usize,
                       n_completion: usize, hit_eos: bool) -> String {
    let id = format!("cmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let choice = Json::Obj(vec![
        ("text".into(),          Json::Str(text.to_string())),
        ("index".into(),         Json::Num(0.0)),
        ("logprobs".into(),      Json::Null),
        ("finish_reason".into(), Json::Str(
            if hit_eos { "stop" } else { "length" }.to_string())),
    ]);
    Json::Obj(vec![
        ("id".into(),      Json::Str(id)),
        ("object".into(),  Json::Str("text_completion".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![choice])),
        ("usage".into(),   Json::Obj(vec![
            ("prompt_tokens".into(),     Json::Num(n_prompt as f64)),
            ("completion_tokens".into(), Json::Num(n_completion as f64)),
            ("total_tokens".into(),      Json::Num((n_prompt + n_completion) as f64)),
        ])),
    ]).to_string()
}

/// OpenAI-shaped `chat.completion` response. Same usage stats as the
/// raw-completion shape, but the choice carries a `message` object
/// instead of a flat `text` field — what every chat SDK expects.
fn chat_completion_response(model: &str, text: &str, n_prompt: usize,
                            n_completion: usize, hit_eos: bool) -> String {
    let id = format!("chatcmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let message = Json::Obj(vec![
        ("role".into(),    Json::Str("assistant".into())),
        ("content".into(), Json::Str(text.to_string())),
    ]);
    let choice = Json::Obj(vec![
        ("index".into(),         Json::Num(0.0)),
        ("message".into(),       message),
        ("finish_reason".into(), Json::Str(
            if hit_eos { "stop" } else { "length" }.to_string())),
    ]);
    Json::Obj(vec![
        ("id".into(),      Json::Str(id)),
        ("object".into(),  Json::Str("chat.completion".into())),
        ("created".into(), Json::Num(unix_now() as f64)),
        ("model".into(),   Json::Str(model.to_string())),
        ("choices".into(), Json::Arr(vec![choice])),
        ("usage".into(),   Json::Obj(vec![
            ("prompt_tokens".into(),     Json::Num(n_prompt as f64)),
            ("completion_tokens".into(), Json::Num(n_completion as f64)),
            ("total_tokens".into(),      Json::Num((n_prompt + n_completion) as f64)),
        ])),
    ]).to_string()
}

fn error_body(message: &str, kind: &str) -> String {
    Json::Obj(vec![
        ("error".into(), Json::Obj(vec![
            ("message".into(), Json::Str(message.to_string())),
            ("type".into(),    Json::Str(kind.to_string())),
        ])),
    ]).to_string()
}

// --- the resident model ------------------------------------------------

/// A loaded GPU model — either runtime family — plus its tokenizer and a
/// reusable decode state. `generate` runs one prompt→completion.
enum ServerModel {
    Qwen {
        gpu: crate::runtime::qwen35::GpuQwen35,
        state: crate::runtime::qwen35::Qwen35GpuState,
        tok: crate::tokenizer::Tokenizer,
        eos: u32,
        max_seq: usize,
        name: String,
    },
    Gemma {
        gpu: crate::runtime::gemma4::GpuGemma4,
        state: crate::runtime::gemma4::Gemma4GpuState,
        tok: crate::tokenizer::GemmaTokenizer,
        eos: u32,
        bos: u32,
        max_seq: usize,
        name: String,
        /// MTP drafter for spec-decode. `None` ⇒ server was started
        /// without `--big-drafter`; every request runs plain decode.
        drafter: Option<GemmaDrafter>,
    },
}

/// MTP drafter resources tied to a Gemma target. Captured graphs
/// (one per K seen) are lazily cached across requests since they're
/// K-shape-specific and reusable from any base_pos.
struct GemmaDrafter {
    runtime: crate::runtime::gemma4_assistant::GpuGemma4Assistant,
    /// `verify_graphs[k]` is a HIP-graph capture of `enqueue_verify_kernels`
    /// for K=k; reused across requests. Index 0 is unused (K must be ≥ 1).
    /// MoE targets keep this empty — they use the decode-loop verify path
    /// inside `verify_forward` (no graph to capture).
    verify_graphs: Vec<Option<crate::hip::GraphExec>>,
}

impl ServerModel {
    /// Load a GGUF, detecting the architecture, into a resident GPU model.
    /// `drafter_path` is honoured only on Gemma 4 targets — qwen35 has no
    /// supported drafter (Qwen 3.6 MTP loads but its forward path is
    /// unwritten; see the gemma4-mtp memory file for the round arithmetic).
    fn load(path: &PathBuf, drafter_path: Option<&PathBuf>, cache: &KernelCache,
            max_seq: usize) -> Result<ServerModel, String>
    {
        let g = GgufFile::open(path).map_err(|e| e.to_string())?;
        let arch = g.metadata_get("general.architecture")
            .and_then(|v| v.as_str()).unwrap_or("<unknown>").to_string();
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();

        // Detect the chat template family from the GGUF's jinja blob;
        // log the family + warn if serve can't apply it natively.
        if let Some(t) = g.metadata_get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
        {
            let fam = crate::chat::detect_chat_template(t);
            if fam.supported_by_serve() {
                eprintln!("[serve]   chat template: {} (supported natively)",
                          fam.label());
            } else {
                eprintln!("[serve]   chat template: {} — NOT applied by serve. \
                           /v1/chat/completions will fail at format time. \
                           Use /v1/completions and pre-template client-side.",
                          fam.label());
            }
        }

        if arch == "gemma4" {
            use crate::model::gemma4::Gemma4Model;
            use crate::model::gemma4_assistant::Gemma4AssistantModel;
            use crate::runtime::gemma4::{GpuGemma4, Gemma4GpuState};
            use crate::runtime::gemma4_assistant::GpuGemma4Assistant;
            use crate::tokenizer::GemmaTokenizer;
            let model = Gemma4Model::load(&g).map_err(|e| e.to_string())?;
            let eos = model.config.eos_token_id;
            let gpu = GpuGemma4::new(&model, &g, cache, max_seq)?;
            let state = Gemma4GpuState::new(&model, max_seq)?;
            let tok = GemmaTokenizer::from_gguf(&g).map_err(|e| e.to_string())?;
            let bos = tok.bos_id;
            let drafter = if let Some(dp) = drafter_path {
                eprintln!("[serve] loading big-drafter   {} ...", dp.display());
                let t = std::time::Instant::now();
                let dg = GgufFile::open(dp).map_err(|e| e.to_string())?;
                let dm = Gemma4AssistantModel::load(&dg).map_err(|e| e.to_string())?;
                let dr = GpuGemma4Assistant::new(&dm, &dg, &gpu, cache)?;
                eprintln!("[serve]   loaded drafter in {:.1}s",
                          t.elapsed().as_secs_f32());
                let mut verify_graphs = Vec::with_capacity(5);
                for _ in 0..5 { verify_graphs.push(None); }
                Some(GemmaDrafter { runtime: dr, verify_graphs })
            } else { None };
            Ok(ServerModel::Gemma { gpu, state, tok, eos, bos, max_seq, name, drafter })
        } else {
            // qwen35 / qwen35moe — the dense + MoE Qwen runtime.
            use crate::model::qwen3_5::Qwen35Model;
            use crate::runtime::qwen35::{GpuQwen35, Qwen35GpuState};
            use crate::tokenizer::Tokenizer;
            let model = Qwen35Model::load(&g).map_err(|e| e.to_string())?;
            let eos = model.config.eos_token_id;
            let gpu = GpuQwen35::new(&model, &g, cache, max_seq)?;
            let state = Qwen35GpuState::new(&model, max_seq)?;
            let tok = Tokenizer::from_gguf(&g)?;
            if drafter_path.is_some() {
                eprintln!("[serve] note: --big-drafter ignored on qwen35 target \
                           (no supported drafter; see gemma4-mtp memory file)");
            }
            Ok(ServerModel::Qwen { gpu, state, tok, eos, max_seq, name })
        }
    }

    fn name(&self) -> &str {
        match self { ServerModel::Qwen { name, .. } | ServerModel::Gemma { name, .. } => name }
    }

    /// Run one completion. Returns (text, prompt_tokens, completion_tokens, hit_eos).
    /// `on_token`, if Some, receives the decoded text DELTA for each
    /// emitted token. Returning `false` from it (e.g. because the
    /// streaming channel closed — the client disconnected) aborts the
    /// generation early; the partial text accumulated so far is still
    /// returned.
    fn generate(&mut self, req: &GenReq,
                mut on_token: impl FnMut(&str) -> bool)
        -> Result<(String, usize, usize, bool), String>
    {
        use crate::sampling::{Rng, sample_chain};
        let mut sp = req.sampler.clone();
        let mut rng = Rng::new(sp.seed);
        // `history` is the decoded-so-far token sequence (for repetition
        // penalty); `counts` is the same data laid out per-vocab for
        // OpenAI-style frequency/presence penalties. Both are empty when
        // the per-request knobs leave their defaults.
        let deadline = req.request_timeout
            .map(|d| std::time::Instant::now() + d);

        match self {
            ServerModel::Qwen { gpu, state, tok, eos, max_seq, .. } => {
                let prompt = match &req.prompt {
                    PromptInput::Raw(text) => tok.encode(text),
                    PromptInput::Chat(msgs) => {
                        // Qwen 3.5/3.6: render via the qwen template
                        // (no BOS — qwen expects to start at <|im_start|>),
                        // with assistant turn primed.
                        crate::chat::format_qwen3(tok, msgs, true)?
                    }
                };
                if prompt.is_empty() {
                    return Err("prompt encoded to zero tokens".into());
                }
                if prompt.len() + req.max_tokens + 4 > *max_seq {
                    return Err(format!(
                        "prompt ({}) + max_tokens ({}) exceeds context window ({})",
                        prompt.len(), req.max_tokens, *max_seq));
                }
                state.reset()?;
                let mut logits = if prompt.len() > 1 {
                    gpu.forward_tokens_batched(&prompt, state)?
                } else {
                    gpu.forward_tokens(&prompt, state)?
                };
                let vocab = logits.len();
                let mut counts: Vec<u16> = if sp.frequency_penalty != 0.0
                    || sp.presence_penalty != 0.0 { vec![0u16; vocab] } else { Vec::new() };
                let mut out: Vec<u32> = Vec::new();
                let mut hit_eos = false;
                let mut prev_text_len: usize = 0;
                let mut full_text = String::new();
                for _ in 0..req.max_tokens {
                    if let Some(d) = deadline {
                        if std::time::Instant::now() >= d { break; }
                    }
                    let t = sample_chain(&mut logits, &mut sp, &out, &counts, &mut rng);
                    if t == *eos { hit_eos = true; break; }
                    out.push(t);
                    if !counts.is_empty() { counts[t as usize] = counts[t as usize].saturating_add(1); }
                    // Re-decode the whole output: append-only token streams
                    // mean the previous prefix bytes are stable, so the
                    // delta is the suffix past prev_text_len. Multi-token
                    // unicode glyphs render correctly because we only emit
                    // bytes once the trailing token completes them.
                    full_text = tok.decode(&out);
                    if full_text.len() > prev_text_len {
                        let delta = &full_text[prev_text_len..];
                        if !on_token(delta) {
                            // Channel closed (client disconnected). Stop
                            // generating; return what we have.
                            break;
                        }
                        prev_text_len = full_text.len();
                    }
                    logits = gpu.forward_token(t, state)?;
                }
                Ok((full_text, prompt.len(), out.len(), hit_eos))
            }
            ServerModel::Gemma { gpu, state, tok, eos, bos, max_seq, drafter, .. } => {
                let prompt = match &req.prompt {
                    PromptInput::Raw(text) => {
                        let mut p = vec![*bos];
                        p.extend(tok.encode(text));
                        p
                    }
                    PromptInput::Chat(msgs) => {
                        // Gemma 4: BOS + per-turn <|turn>role\n…<turn|>\n
                        // with assistant turn primed. The drafter was
                        // trained on this format — chat-templated input
                        // typically gets +30-40 percentage points of
                        // accept rate over raw user text.
                        crate::chat::format_gemma4(tok, msgs, true)?
                    }
                };
                if prompt.is_empty() {
                    return Err("prompt encoded to zero tokens".into());
                }
                if prompt.len() + req.max_tokens + 8 > *max_seq {
                    return Err(format!(
                        "prompt ({}) + max_tokens ({}) exceeds context window ({})",
                        prompt.len(), req.max_tokens, *max_seq));
                }
                // Dispatch: spec-decode when a drafter is loaded AND the
                // request hasn't opted out. Default-on if drafter present.
                let want_spec = match req.use_speculative {
                    Some(b) => b,
                    None    => drafter.is_some(),
                };
                let do_spec = want_spec && drafter.is_some();
                if want_spec && drafter.is_none() {
                    return Err("use_speculative=true but server has no drafter loaded \
                                (start with --big-drafter PATH)".into());
                }

                state.reset();

                if !do_spec {
                    // Plain prefill + decode.
                    let mut logits = gpu.prefill_forward(&prompt, state)?;
                    let vocab = logits.len();
                    let mut counts: Vec<u16> = if sp.frequency_penalty != 0.0
                        || sp.presence_penalty != 0.0 { vec![0u16; vocab] } else { Vec::new() };
                    let mut out: Vec<u32> = Vec::new();
                    let mut hit_eos = false;
                    let mut prev_text_len: usize = 0;
                    let mut full_text = String::new();
                    for _ in 0..req.max_tokens {
                        if let Some(d) = deadline {
                            if std::time::Instant::now() >= d { break; }
                        }
                        let t = sample_chain(&mut logits, &mut sp, &out, &counts, &mut rng);
                        if t == *eos { hit_eos = true; break; }
                        out.push(t);
                        if !counts.is_empty() {
                            counts[t as usize] = counts[t as usize].saturating_add(1);
                        }
                        full_text = tok.decode(&out);
                        if full_text.len() > prev_text_len {
                            let delta = &full_text[prev_text_len..];
                            if !on_token(delta) { break; }
                            prev_text_len = full_text.len();
                        }
                        logits = gpu.forward_token(t, state)?;
                    }
                    return Ok((full_text, prompt.len(), out.len(), hit_eos));
                }

                // Spec-decode path: prefill, then K=req.speculative_k
                // (default 3) rounds via the shared loop. Verify graphs
                // are captured lazily per K and cached across requests.
                let d = drafter.as_mut().unwrap();
                let k = req.speculative_k.unwrap_or(3).clamp(1, 4);
                // Prefill all but the last token — its logits aren't
                // useful; the verify path immediately re-forwards it
                // through `forward_token` to seed the chain.
                let _ = gpu.prefill_forward(&prompt[..prompt.len() - 1], state)?;
                let verify_logits = gpu.forward_token(*prompt.last().unwrap(), state)?;
                if d.verify_graphs[k].is_none() && !gpu.is_moe() {
                    d.verify_graphs[k] = Some(gpu.capture_verify_graph(state, k)?);
                }
                let (gen_toks, stats) = crate::runtime::spec_decode::spec_decode_generate(
                    gpu, &d.runtime, state,
                    d.verify_graphs[k].as_ref(), k,
                    verify_logits,
                    *prompt.last().unwrap(),
                    *eos,
                    req.max_tokens, k, req.sampler.temperature, req.sampler.seed,
                )?;
                eprintln!("[serve] spec-decode K={k}: {}/{} accept ({:.0}%)",
                    stats.n_accepted, stats.n_drafted, 100.0 * stats.accept_rate());
                Ok((tok.decode(&gen_toks), prompt.len(), gen_toks.len(), stats.hit_eos))
            }
        }
    }
}

// --- the GPU worker ----------------------------------------------------

fn worker(rx: mpsc::Receiver<Job>, big: PathBuf, big_drafter: Option<PathBuf>,
          small: PathBuf, max_seq: usize, metrics: Arc<Metrics>)
{
    let setup = (|| -> Result<(KernelCache, ServerModel, ServerModel), String> {
        crate::hip::Device::set(0)?;
        let cache = KernelCache::new()?;
        let load = |label: &str, path: &PathBuf, drafter: Option<&PathBuf>|
            -> Result<ServerModel, String>
        {
            eprintln!("[serve] loading {label:5} model {} ...", path.display());
            let t = std::time::Instant::now();
            let m = ServerModel::load(path, drafter, &cache, max_seq)
                .map_err(|e| {
                    // VRAM-exhaustion → add a hint about model size vs VRAM.
                    if e.to_lowercase().contains("memory") {
                        let sz = std::fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
                        format!("{label} model: {e}\n\
                                 [hint] model file is {:.1} GB on disk; \
                                 with KV cache for max_seq={max_seq} the GPU needs roughly \
                                 1.2-1.5× that. Check `rocm-smi --showmeminfo vram` \
                                 against your model's expected resident size, or lower \
                                 max_seq with --max-seq.",
                                 sz as f64 / (1024.0 * 1024.0 * 1024.0))
                    } else {
                        format!("{label} model: {e}")
                    }
                })?;
            eprintln!("[serve]   loaded {} in {:.1}s", m.name(), t.elapsed().as_secs_f32());
            Ok(m)
        };
        let big_m   = load("big",   &big,   big_drafter.as_ref())?;
        let small_m = load("small", &small, None)?;
        Ok((cache, big_m, small_m))
    })();

    let (_cache, mut big_m, mut small_m) = match setup {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[serve] FATAL: model load failed: {e}");
            // Drain the queue with 503s so clients don't hang forever.
            for job in rx {
                let _ = job.reply.send(StreamMsg::Done(HttpReply {
                    status: 503, status_text: "Service Unavailable",
                    body: error_body(&format!("model load failed: {e}"), "server_error"),
                }));
            }
            return;
        }
    };
    eprintln!("[serve] ready — serving requests.");

    for job in rx {
        let reply = match job.req {
            Err((status, status_text, msg)) => {
                eprintln!("[serve] req={} target={} status={} reason={:?} msg={}",
                          job.request_id, job.target.label(), status, status_text, msg);
                metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
                HttpReply {
                    status, status_text, body: error_body(&msg, "invalid_request_error"),
                }
            }
            Ok(req) => match job.target {
                Target::Embed => {
                    eprintln!("[serve] req={} target=embed status=503 reason=not-yet-available",
                              job.request_id);
                    metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
                    HttpReply {
                        status: 503, status_text: "Service Unavailable",
                        body: error_body(
                            "embedder not yet available — nomic-bert encoder is a follow-up",
                            "server_error"),
                    }
                }
                Target::Big | Target::Small => {
                    let model = if job.target == Target::Big { &mut big_m } else { &mut small_m };
                    let t = std::time::Instant::now();
                    let is_chat = req.is_chat();
                    let is_stream = req.stream;
                    // For streaming requests, build a one-shot SSE id +
                    // first-chunk role frame (chat) up front so the
                    // per-token callback can emit just text deltas.
                    let stream_id = if is_chat {
                        format!("chatcmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed))
                    } else {
                        format!("cmpl-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed))
                    };
                    let model_name = model.name().to_string();
                    let reply_tx = job.reply.clone();
                    // For chat streams, emit a role-only opener frame
                    // before any text content (matches OpenAI SDK
                    // expectations).
                    if is_stream && is_chat {
                        let frame = chat_stream_chunk(&stream_id, &model_name,
                            ChatDelta { role: Some("assistant"), content: None }, None);
                        let _ = reply_tx.send(StreamMsg::Chunk(frame));
                    }
                    // `catch_unwind` around generate(): a panic in a kernel
                    // launch, a slipped unwrap, or a numerical blowup
                    // shouldn't kill the worker thread (which would 503
                    // every subsequent request). On panic we 500 the
                    // current request, log, and continue.
                    let stream_id_for_cb = stream_id.clone();
                    let model_name_for_cb = model_name.clone();
                    let reply_for_cb = reply_tx.clone();
                    let on_token = move |delta: &str| -> bool {
                        if !is_stream { return true; }
                        let frame = if is_chat {
                            chat_stream_chunk(&stream_id_for_cb, &model_name_for_cb,
                                ChatDelta { role: None, content: Some(delta) }, None)
                        } else {
                            completion_stream_chunk(&stream_id_for_cb, &model_name_for_cb,
                                delta, None)
                        };
                        reply_for_cb.send(StreamMsg::Chunk(frame)).is_ok()
                    };
                    let result = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| model.generate(&req, on_token)));
                    match result {
                        Ok(Ok((text, n_p, n_c, eos))) => {
                            let wall_us = t.elapsed().as_micros() as u64;
                            metrics.requests_ok.fetch_add(1, Ordering::Relaxed);
                            metrics.prompt_tokens.fetch_add(n_p as u64, Ordering::Relaxed);
                            metrics.completion_tokens.fetch_add(n_c as u64, Ordering::Relaxed);
                            metrics.decode_us_total.fetch_add(wall_us, Ordering::Relaxed);
                            if eos { metrics.requests_eos.fetch_add(1, Ordering::Relaxed); }
                            else   { metrics.requests_length.fetch_add(1, Ordering::Relaxed); }
                            let tok_per_s = if n_c > 0 && wall_us > 0 {
                                n_c as f64 * 1_000_000.0 / wall_us as f64
                            } else { 0.0 };
                            eprintln!("[serve] req={} target={} type={} status=200 \
                                       n_p={} n_c={} wall_ms={:.1} tok_s={:.1} finish={} stream={}",
                                job.request_id, job.target.label(),
                                if is_chat { "chat" } else { "completion" },
                                n_p, n_c, wall_us as f64 / 1000.0, tok_per_s,
                                if eos { "stop" } else { "length" }, is_stream);
                            if is_stream {
                                // Final SSE frame: empty delta + finish_reason.
                                // Then Done signals the connection handler
                                // to write "data: [DONE]\n\n" and close.
                                let fin = if eos { "stop" } else { "length" };
                                let frame = if is_chat {
                                    chat_stream_chunk(&stream_id, &model_name,
                                        ChatDelta { role: None, content: None }, Some(fin))
                                } else {
                                    completion_stream_chunk(&stream_id, &model_name,
                                        "", Some(fin))
                                };
                                let _ = reply_tx.send(StreamMsg::Chunk(frame));
                                HttpReply { status: 200, status_text: "OK",
                                            body: String::new() }
                            } else {
                                let body = if is_chat {
                                    chat_completion_response(&model_name, &text, n_p, n_c, eos)
                                } else {
                                    completion_response(&model_name, &text, n_p, n_c, eos)
                                };
                                HttpReply { status: 200, status_text: "OK", body }
                            }
                        }
                        Ok(Err(e)) => {
                            eprintln!("[serve] req={} target={} status=400 reason={:?}",
                                      job.request_id, job.target.label(), e);
                            metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
                            HttpReply {
                                status: 400, status_text: "Bad Request",
                                body: error_body(&e, "invalid_request_error"),
                            }
                        }
                        Err(payload) => {
                            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                                (*s).to_string()
                            } else if let Some(s) = payload.downcast_ref::<String>() {
                                s.clone()
                            } else {
                                "unknown panic in generate()".to_string()
                            };
                            eprintln!("[serve] req={} target={} status=500 PANIC={}",
                                      job.request_id, job.target.label(), msg);
                            metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
                            metrics.panics_recovered.fetch_add(1, Ordering::Relaxed);
                            HttpReply {
                                status: 500, status_text: "Internal Server Error",
                                body: error_body(
                                    &format!("internal panic: {msg}"),
                                    "server_error"),
                            }
                        }
                    }
                }
            },
        };
        let _ = job.reply.send(StreamMsg::Done(reply));
    }
}

// --- connection handling ----------------------------------------------

fn handle_conn(mut stream: std::net::TcpStream, target: Target,
               tx: mpsc::Sender<Job>, metrics: Arc<Metrics>)
{
    let request_id = metrics.requests_total.fetch_add(1, Ordering::Relaxed) + 1;
    let request = match http::read_request(&stream) {
        Ok(r) => r,
        Err(e) => {
            metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
            eprintln!("[serve] req={request_id} target={} status=400 reason=malformed-http err={e}",
                      target.label());
            let _ = http::write_response(&mut stream, 400, "Bad Request",
                &error_body(&format!("malformed HTTP request: {e}"), "invalid_request_error"));
            return;
        }
    };

    // Plain GET /metrics on any port — serves Prometheus text. Cheap;
    // no GPU work. Operator point-of-entry for serving observability.
    let path = request.path.trim_end_matches('/');
    let is_get = request.method.eq_ignore_ascii_case("GET");
    if is_get && path.ends_with("/metrics") {
        let body = metrics.render_prometheus();
        // Direct write — bypass JSON error_body shape.
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body);
        let _ = std::io::Write::write_all(&mut stream, resp.as_bytes());
        return;
    }
    // Plain GET /healthz — tiny liveness check.
    if is_get && path.ends_with("/healthz") {
        let body = "ok\n";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body);
        let _ = std::io::Write::write_all(&mut stream, resp.as_bytes());
        return;
    }

    // Route. LLM ports take /v1/completions (raw) or /v1/chat/completions
    // (messages, chat template applied server-side). Embed port takes
    // /v1/embeddings (answers 503 until the encoder lands).
    let is_post = request.method.eq_ignore_ascii_case("POST");
    let route = if target == Target::Embed {
        if is_post && path.ends_with("/v1/embeddings") { Some("embed") } else { None }
    } else if is_post && path.ends_with("/v1/chat/completions") {
        Some("chat")
    } else if is_post && path.ends_with("/v1/completions") {
        Some("completions")
    } else {
        None
    };

    let req = match route {
        None => Err((404u16, "Not Found",
            format!("no route for {} {} (expected POST /v1/completions or \
                     /v1/chat/completions on this port, or GET /metrics / /healthz)",
                    request.method, request.path))),
        Some("embed") => {
            // Worker answers 503; keep the shape.
            Ok(GenReq {
                prompt: PromptInput::Raw(String::new()),
                max_tokens: 0,
                sampler: crate::sampling::SamplerParams::default(),
                use_speculative: None,
                speculative_k: None,
                request_timeout: None,
                stream: false,
            })
        }
        Some("chat") => parse_chat_completions(&request.body),
        Some("completions") => parse_completions(&request.body),
        Some(other) => unreachable!("unknown route tag {other}"),
    };

    let (rtx, rrx) = mpsc::channel();
    if tx.send(Job { request_id, target, req, reply: rtx }).is_err() {
        metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
        let _ = http::write_response(&mut stream, 503, "Service Unavailable",
            &error_body("server worker is gone", "server_error"));
        return;
    }
    // Receive the first message: if it's Done, plain response. If it's
    // Chunk, switch to SSE streaming mode (writing each chunk as it arrives
    // until a Done arrives or the channel closes).
    use std::io::Write;
    let first = rrx.recv();
    match first {
        Ok(StreamMsg::Done(reply)) => {
            let _ = http::write_response(&mut stream, reply.status, reply.status_text, &reply.body);
        }
        Ok(StreamMsg::Chunk(payload)) => {
            // SSE: open with the appropriate headers, then write each
            // chunk as `data: {payload}\n\n`. Final `data: [DONE]\n\n`
            // matches OpenAI's terminator. If a socket write fails
            // mid-stream, drop the connection — the worker will see the
            // channel close on its next send and stop generating.
            let header = "HTTP/1.1 200 OK\r\n\
                          Content-Type: text/event-stream\r\n\
                          Cache-Control: no-cache\r\n\
                          Connection: close\r\n\r\n";
            if stream.write_all(header.as_bytes()).is_err() { return; }
            if stream.write_all(format!("data: {payload}\n\n").as_bytes()).is_err() { return; }
            let _ = stream.flush();
            loop {
                match rrx.recv() {
                    Ok(StreamMsg::Chunk(p)) => {
                        if stream.write_all(format!("data: {p}\n\n").as_bytes()).is_err() { return; }
                        let _ = stream.flush();
                    }
                    Ok(StreamMsg::Done(_reply)) => {
                        // Standard OpenAI SSE terminator. The summary
                        // reply body isn't sent in streaming mode —
                        // clients accumulate the chunks.
                        let _ = stream.write_all(b"data: [DONE]\n\n");
                        let _ = stream.flush();
                        return;
                    }
                    Err(_) => {
                        // Worker dropped the channel. Close stream.
                        return;
                    }
                }
            }
        }
        Err(_) => {
            metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
            let _ = http::write_response(&mut stream, 500, "Internal Server Error",
                &error_body("worker dropped the request", "server_error"));
        }
    }
}

fn acceptor(port: u16, target: Target, tx: mpsc::Sender<Job>, metrics: Arc<Metrics>) {
    let listener = match std::net::TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => { eprintln!("[serve] FATAL: cannot bind port {port}: {e}"); return; }
    };
    eprintln!("[serve] {} listening on :{port}", target.label());
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let tx = tx.clone();
                let metrics = Arc::clone(&metrics);
                thread::spawn(move || handle_conn(stream, target, tx, metrics));
            }
            Err(e) => eprintln!("[serve] accept error on :{port}: {e}"),
        }
    }
}

/// Start the three-port multi-model server. Blocks forever.
pub fn run(big: PathBuf, big_drafter: Option<PathBuf>,
           small: PathBuf, embed: Option<PathBuf>,
           big_port: u16, small_port: u16, embed_port: u16, max_seq: usize)
    -> Result<(), String>
{
    // Surface any REINSTINCT_* env vars at startup. Several of them are
    // perf-killers if set unintentionally on a serve box (graph capture
    // off, dp4a path off, etc) — better to log them than have an
    // operator chasing a silent regression weeks later.
    let env_warnings: Vec<(&str, bool)> = vec![
        ("REINSTINCT_NO_GRAPH",        true),
        ("REINSTINCT_MOE_PROFILE",     true),
        ("REINSTINCT_NO_DP4A_Q4",      true),
        ("REINSTINCT_NO_DP4A_Q5",      true),
        ("REINSTINCT_NO_DP4A_Q6",      true),
        ("REINSTINCT_NO_DP4A_Q8",      true),
        ("REINSTINCT_GEMMA_NO_DP4A",   true),
        ("REINSTINCT_GDN_NO_LDS128",   true),
        ("REINSTINCT_OLD_ATTN",        true),
        ("REINSTINCT_PREFILL_NO_GRAPH", true),
        ("REINSTINCT_PREFILL_DEBUG",   false),
        ("REINSTINCT_DECODE_DEBUG",    false),
        ("REINSTINCT_PREFILL_TRACE",   false),
    ];
    for (var, is_perf_killer) in &env_warnings {
        if std::env::var_os(var).is_some() {
            let tag = if *is_perf_killer { "WARN" } else { "info" };
            eprintln!("[serve] {tag}: {var} is set — \
                       {}", if *is_perf_killer {
                "perf will regress significantly; unset for production"
            } else {
                "tracing on; expect verbose logs"
            });
        }
    }

    if let Some(e) = &embed {
        eprintln!("[serve] note: --embed {} accepted but deferred — \
                   the :{embed_port} port will answer 503 until the \
                   nomic-bert encoder lands.", e.display());
    }

    let (tx, rx) = mpsc::channel::<Job>();
    let metrics = Arc::new(Metrics::new());

    let worker_handle = {
        let (big, big_drafter, small) =
            (big.clone(), big_drafter.clone(), small.clone());
        let metrics = Arc::clone(&metrics);
        thread::Builder::new().name("gpu-worker".into())
            .spawn(move || worker(rx, big, big_drafter, small, max_seq, metrics))
            .map_err(|e| e.to_string())?
    };

    for (port, target) in [(big_port, Target::Big),
                           (small_port, Target::Small),
                           (embed_port, Target::Embed)] {
        let tx = tx.clone();
        let metrics = Arc::clone(&metrics);
        thread::Builder::new().name(format!("accept-{}", target.label()))
            .spawn(move || acceptor(port, target, tx, metrics))
            .map_err(|e| e.to_string())?;
    }
    drop(tx);   // only the acceptors hold senders now

    worker_handle.join().map_err(|_| "gpu worker panicked".to_string())?;
    Ok(())
}
