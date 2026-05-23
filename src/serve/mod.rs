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
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use json::Json;

use crate::gguf::GgufFile;
use crate::runtime::KernelCache;

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
    temperature: f32,
    top_k: usize,
    seed: u64,
    /// MTP spec-decode opt-in/opt-out. `None` ⇒ use the server default
    /// (true if the target has a drafter loaded, false otherwise).
    /// `Some(false)` lets a per-turn classifier disable the drafter on
    /// creative work where it would just waste verify cycles.
    use_speculative: Option<bool>,
    /// Per-request K override for spec-decode. `None` ⇒ server default
    /// (currently 3). Ignored when spec-decode is off.
    speculative_k: Option<usize>,
}

impl GenReq {
    fn is_chat(&self) -> bool { matches!(self.prompt, PromptInput::Chat(_)) }
}

/// A unit of work handed from a connection thread to the GPU worker.
struct Job {
    target: Target,
    /// `Err` carries an already-formed client error (bad request / wrong route).
    req: Result<GenReq, (u16, &'static str, String)>,
    reply: mpsc::Sender<HttpReply>,
}

struct HttpReply {
    status: u16,
    status_text: &'static str,
    body: String,
}

// --- request parsing ---------------------------------------------------

/// Common decode + sampling fields shared by both endpoint parsers.
/// Returns (max_tokens, temperature, top_k, seed, use_speculative, speculative_k).
fn parse_common_fields(j: &Json) -> (usize, f32, usize, u64, Option<bool>, Option<usize>) {
    let max_tokens = j.get("max_tokens").and_then(Json::as_f64)
        .map(|n| n as usize).unwrap_or(256).clamp(1, 4096);
    let temperature = j.get("temperature").and_then(Json::as_f64)
        .map(|n| n as f32).unwrap_or(0.8).max(0.0);
    let top_k = j.get("top_k").and_then(Json::as_f64)
        .map(|n| n as usize).unwrap_or(40);
    let seed = j.get("seed").and_then(Json::as_f64)
        .map(|n| n as u64).unwrap_or(0);
    let use_speculative = j.get("use_speculative").and_then(Json::as_bool);
    let speculative_k = j.get("speculative_k").and_then(Json::as_f64)
        .map(|n| (n as usize).clamp(1, 4));
    (max_tokens, temperature, top_k, seed, use_speculative, speculative_k)
}

/// Parse an OpenAI `/v1/completions` body into a `GenReq`. Raw-prompt
/// path; no chat template is applied server-side.
fn parse_completions(body: &str) -> Result<GenReq, (u16, &'static str, String)> {
    let bad = |m: String| (400u16, "Bad Request", m);
    let j = Json::parse(body).map_err(|e| bad(format!("invalid JSON: {e}")))?;
    let prompt = j.get("prompt").and_then(Json::as_str)
        .ok_or_else(|| bad("missing string field 'prompt'".into()))?
        .to_string();
    let (max_tokens, temperature, top_k, seed, use_speculative, speculative_k)
        = parse_common_fields(&j);
    Ok(GenReq { prompt: PromptInput::Raw(prompt), max_tokens, temperature, top_k,
                seed, use_speculative, speculative_k })
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
    let (max_tokens, temperature, top_k, seed, use_speculative, speculative_k)
        = parse_common_fields(&j);
    Ok(GenReq { prompt: PromptInput::Chat(messages), max_tokens, temperature, top_k,
                seed, use_speculative, speculative_k })
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
    fn generate(&mut self, req: &GenReq)
        -> Result<(String, usize, usize, bool), String>
    {
        use crate::sampling::{Rng, sample_temp_topk};

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
                let mut rng = Rng::new(req.seed);
                let mut out = Vec::new();
                let mut hit_eos = false;
                for _ in 0..req.max_tokens {
                    let t = sample_temp_topk(&logits, req.temperature, req.top_k, &mut rng);
                    if t == *eos { hit_eos = true; break; }
                    out.push(t);
                    logits = gpu.forward_token(t, state)?;
                }
                Ok((tok.decode(&out), prompt.len(), out.len(), hit_eos))
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
                    let mut rng = Rng::new(req.seed);
                    let mut out = Vec::new();
                    let mut hit_eos = false;
                    for _ in 0..req.max_tokens {
                        let t = sample_temp_topk(&logits, req.temperature, req.top_k, &mut rng);
                        if t == *eos { hit_eos = true; break; }
                        out.push(t);
                        logits = gpu.forward_token(t, state)?;
                    }
                    return Ok((tok.decode(&out), prompt.len(), out.len(), hit_eos));
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
                    req.max_tokens, k, req.temperature, req.seed,
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
          small: PathBuf, max_seq: usize)
{
    let setup = (|| -> Result<(KernelCache, ServerModel, ServerModel), String> {
        crate::hip::Device::set(0)?;
        let cache = KernelCache::new()?;
        eprintln!("[serve] loading big model   {} ...", big.display());
        let t = std::time::Instant::now();
        let big_m = ServerModel::load(&big, big_drafter.as_ref(), &cache, max_seq)?;
        eprintln!("[serve]   loaded {} in {:.1}s", big_m.name(), t.elapsed().as_secs_f32());
        eprintln!("[serve] loading small model {} ...", small.display());
        let t = std::time::Instant::now();
        let small_m = ServerModel::load(&small, None, &cache, max_seq)?;
        eprintln!("[serve]   loaded {} in {:.1}s", small_m.name(), t.elapsed().as_secs_f32());
        Ok((cache, big_m, small_m))
    })();

    let (_cache, mut big_m, mut small_m) = match setup {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[serve] FATAL: model load failed: {e}");
            // Drain the queue with 503s so clients don't hang forever.
            for job in rx {
                let _ = job.reply.send(HttpReply {
                    status: 503, status_text: "Service Unavailable",
                    body: error_body(&format!("model load failed: {e}"), "server_error"),
                });
            }
            return;
        }
    };
    eprintln!("[serve] ready — serving requests.");

    for job in rx {
        let reply = match job.req {
            Err((status, status_text, msg)) => HttpReply {
                status, status_text, body: error_body(&msg, "invalid_request_error"),
            },
            Ok(req) => match job.target {
                Target::Embed => HttpReply {
                    status: 503, status_text: "Service Unavailable",
                    body: error_body(
                        "embedder not yet available — nomic-bert encoder is a follow-up",
                        "server_error"),
                },
                Target::Big | Target::Small => {
                    let model = if job.target == Target::Big { &mut big_m } else { &mut small_m };
                    let t = std::time::Instant::now();
                    let is_chat = req.is_chat();
                    match model.generate(&req) {
                        Ok((text, n_p, n_c, eos)) => {
                            eprintln!("[serve] {} {}: {} prompt + {} gen tok in {:.2}s",
                                job.target.label(),
                                if is_chat { "chat" } else { "completion" },
                                n_p, n_c, t.elapsed().as_secs_f32());
                            let body = if is_chat {
                                chat_completion_response(model.name(), &text, n_p, n_c, eos)
                            } else {
                                completion_response(model.name(), &text, n_p, n_c, eos)
                            };
                            HttpReply { status: 200, status_text: "OK", body }
                        }
                        Err(e) => HttpReply {
                            status: 400, status_text: "Bad Request",
                            body: error_body(&e, "invalid_request_error"),
                        },
                    }
                }
            },
        };
        let _ = job.reply.send(reply);
    }
}

// --- connection handling ----------------------------------------------

fn handle_conn(mut stream: std::net::TcpStream, target: Target, tx: mpsc::Sender<Job>) {
    let request = match http::read_request(&stream) {
        Ok(r) => r,
        Err(e) => {
            let _ = http::write_response(&mut stream, 400, "Bad Request",
                &error_body(&format!("malformed HTTP request: {e}"), "invalid_request_error"));
            return;
        }
    };

    // Route. LLM ports take /v1/completions (raw) or /v1/chat/completions
    // (messages, chat template applied server-side). Embed port takes
    // /v1/embeddings (answers 503 until the encoder lands).
    let path = request.path.trim_end_matches('/');
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
                     /v1/chat/completions on this port)",
                    request.method, request.path))),
        Some("embed") => {
            // Worker answers 503; keep the shape.
            Ok(GenReq { prompt: PromptInput::Raw(String::new()), max_tokens: 0,
                        temperature: 0.0, top_k: 0, seed: 0,
                        use_speculative: None, speculative_k: None })
        }
        Some("chat") => parse_chat_completions(&request.body),
        Some("completions") => parse_completions(&request.body),
        Some(other) => unreachable!("unknown route tag {other}"),
    };

    let (rtx, rrx) = mpsc::channel();
    if tx.send(Job { target, req, reply: rtx }).is_err() {
        let _ = http::write_response(&mut stream, 503, "Service Unavailable",
            &error_body("server worker is gone", "server_error"));
        return;
    }
    let reply = rrx.recv().unwrap_or(HttpReply {
        status: 500, status_text: "Internal Server Error",
        body: error_body("worker dropped the request", "server_error"),
    });
    let _ = http::write_response(&mut stream, reply.status, reply.status_text, &reply.body);
}

fn acceptor(port: u16, target: Target, tx: mpsc::Sender<Job>) {
    let listener = match std::net::TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => { eprintln!("[serve] FATAL: cannot bind port {port}: {e}"); return; }
    };
    eprintln!("[serve] {} listening on :{port}", target.label());
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let tx = tx.clone();
                thread::spawn(move || handle_conn(stream, target, tx));
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
    if let Some(e) = &embed {
        eprintln!("[serve] note: --embed {} accepted but deferred — \
                   the :{embed_port} port will answer 503 until the \
                   nomic-bert encoder lands.", e.display());
    }

    let (tx, rx) = mpsc::channel::<Job>();

    let worker_handle = {
        let (big, big_drafter, small) =
            (big.clone(), big_drafter.clone(), small.clone());
        thread::Builder::new().name("gpu-worker".into())
            .spawn(move || worker(rx, big, big_drafter, small, max_seq))
            .map_err(|e| e.to_string())?
    };

    for (port, target) in [(big_port, Target::Big),
                           (small_port, Target::Small),
                           (embed_port, Target::Embed)] {
        let tx = tx.clone();
        thread::Builder::new().name(format!("accept-{}", target.label()))
            .spawn(move || acceptor(port, target, tx))
            .map_err(|e| e.to_string())?;
    }
    drop(tx);   // only the acceptors hold senders now

    worker_handle.join().map_err(|_| "gpu worker panicked".to_string())?;
    Ok(())
}
