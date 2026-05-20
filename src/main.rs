use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use reinstinct_engine::cpu::qwen3_5::{ForwardTrace, Qwen35F32Model};
use reinstinct_engine::gguf::{GgufFile, MetaValue};
use reinstinct_engine::model::qwen3_5::{BlockKind, Qwen35Model};

#[derive(Parser, Debug)]
#[command(name = "reinstinct-engine", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print header, metadata, and a tensor-type histogram for a GGUF file.
    Inspect {
        path: PathBuf,
        /// Show every tensor (default: histogram + top-10 by size).
        #[arg(long)]
        verbose: bool,
    },
    /// Detect architecture and parse as a typed model (currently: qwen35 only).
    Model {
        path: PathBuf,
    },
    /// Run the forward pass on one or more input tokens, print top-K logits.
    Generate {
        path: PathBuf,
        /// Input token id. Defaults to the model's EOS token id from metadata.
        /// Ignored if --tokens is provided.
        #[arg(short, long)]
        token: Option<u32>,
        /// Comma-separated list of token ids to feed in order. Logits are
        /// printed for the LAST position. Overrides --token.
        #[arg(long, value_delimiter = ',')]
        tokens: Option<Vec<u32>>,
        /// Number of top logits to print.
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// Run on the GPU (HIP) instead of the CPU oracle.
        #[arg(long)]
        gpu: bool,
    },
    /// Sample tokens autoregressively from a prompt — single token or
    /// comma-separated prefill, then `--steps` newly generated tokens.
    GenerateText {
        path: PathBuf,
        /// Prompt as text — encoded via the GGUF BPE tokenizer.
        /// Takes precedence over --tokens.
        #[arg(long)]
        prompt: Option<String>,
        /// Gemma 4 chat-template system message (rendered with the model's
        /// chat template; only `gemma4` models). Overrides --prompt when set.
        #[arg(long)]
        system: Option<String>,
        /// Gemma 4 chat-template user message (paired with --system). Falls
        /// back to --prompt's content if not given.
        #[arg(long)]
        user: Option<String>,
        /// Comma-separated prompt tokens. Defaults to [eos_token_id].
        #[arg(long, value_delimiter = ',')]
        tokens: Option<Vec<u32>>,
        /// Number of new tokens to sample after the prompt is consumed.
        #[arg(short = 'n', long, default_value_t = 32)]
        steps: usize,
        /// Sampling temperature (0 = greedy/argmax).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Top-k filter (0 = no filter, full vocab).
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// PRNG seed.
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
        /// Run on GPU.
        #[arg(long)]
        gpu: bool,
    },
    /// Speculative decode against a Gemma 4 target using its MTP drafter.
    /// Currently sequential-verify (correctness, no speedup) — proves the
    /// accept/reject loop end-to-end; the batched-verify perf win lands
    /// when prefill_forward grows incremental positions.
    MtpGen {
        target: PathBuf,
        drafter: PathBuf,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        system: Option<String>,
        /// Drafted tokens per spec-decode round.
        #[arg(long, default_value_t = 4)]
        k: usize,
        /// Total tokens to generate (round trip until this many accepted).
        #[arg(short = 'n', long, default_value_t = 64)]
        steps: usize,
        /// Sampling temperature. `0` = greedy (strict argmax match accept).
        /// `> 0` switches to rejection-sampling acceptance (accept with
        /// probability min(1, p_target/p_draft); residual sample on reject).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
    },
    /// Spec-decode smoke test: load a Gemma 4 target + its MTP drafter,
    /// prefill a prompt, then ask the drafter to propose K tokens at the
    /// prompt's last position. Prints each drafted token plus the
    /// target's own next-token prediction so the two can be compared.
    /// First-cut diagnostic for the MTP drafter — no acceptance loop /
    /// KV truncate / speedup yet.
    MtpDraft {
        target: PathBuf,
        drafter: PathBuf,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 4)]
        k: usize,
    },
    /// Multi-turn chat against a Gemma 4 model with KV-cache prefix
    /// reuse: the system message is prefilled and snapshotted once,
    /// then each `--turn` reuses the snapshot — TTFT drops to the
    /// per-turn token count instead of the full conversation.
    Chat {
        path: PathBuf,
        /// Gemma 4 system message (rendered with the chat template).
        #[arg(long)]
        system: Option<String>,
        /// Per-turn user input. Pass multiple times to demonstrate
        /// prefix reuse — turn 2+ restores from the snapshot rather
        /// than re-prefilling the system.
        #[arg(long = "turn")]
        turns: Vec<String>,
        /// Decode tokens per turn.
        #[arg(short = 'n', long, default_value_t = 60)]
        steps: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
    },
    /// Dump diagnostic stats for the embedding row of one or more tokens.
    DebugEmbed {
        path: PathBuf,
        tokens: Vec<u32>,
    },
    /// Run forward N times, report per-stage timing breakdown.
    Bench {
        path: PathBuf,
        #[arg(short = 'n', long, default_value_t = 5)]
        iters: usize,
        #[arg(short, long)]
        token: Option<u32>,
    },
    /// Print HIP devices, VRAM, and time a host↔device round-trip.
    HipInfo {
        /// MB per copy direction in the bandwidth probe.
        #[arg(long, default_value_t = 64)]
        mb: usize,
        /// Round-trip iterations to average bandwidth over.
        #[arg(long, default_value_t = 8)]
        iters: usize,
    },
    /// Time `forward_token` on the GPU and compare to the CPU baseline.
    GpuBench {
        path: PathBuf,
        #[arg(short = 'n', long, default_value_t = 20)]
        iters: usize,
        #[arg(short, long)]
        token: Option<u32>,
    },
    /// Run a multi-model HTTP server: Big LLM, Small LLM, and Embedder,
    /// each on its own port, requests served in order through one GPU.
    Serve {
        /// Big model GGUF (~30B dense — Qwen 3.x or Gemma 4 31B).
        #[arg(long)]
        big: PathBuf,
        /// Small model GGUF (Qwen 3.5 4B or Gemma E4B).
        #[arg(long)]
        small: PathBuf,
        /// Embedder GGUF (nomic-embed). Accepted but deferred — its
        /// port answers 503 until the encoder runtime lands.
        #[arg(long)]
        embed: Option<PathBuf>,
        #[arg(long, default_value_t = 8080)]
        big_port: u16,
        #[arg(long, default_value_t = 8081)]
        small_port: u16,
        #[arg(long, default_value_t = 8082)]
        embed_port: u16,
        /// Context window (prompt + generated tokens) per request.
        #[arg(long, default_value_t = 4096)]
        max_seq: usize,
    },
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Command::Inspect { path, verbose } => inspect(&path, verbose),
        Command::Model { path } => model(&path),
        Command::Generate { path, token, tokens, k, gpu } => generate(&path, token, tokens, k, gpu),
        Command::DebugEmbed { path, tokens } => debug_embed(&path, &tokens),
        Command::Bench { path, iters, token } => bench(&path, iters, token),
        Command::HipInfo { mb, iters } => hip_info(mb, iters),
        Command::GpuBench { path, iters, token } => gpu_bench(&path, iters, token),
        Command::Serve { big, small, embed, big_port, small_port, embed_port, max_seq } =>
            reinstinct_engine::serve::run(big, small, embed, big_port, small_port,
                                          embed_port, max_seq)
                .map_err(anyhow::Error::msg),
        Command::GenerateText { path, prompt, system, user, tokens, steps,
                                temperature, top_k, seed, gpu } =>
            generate_text(&path, prompt, system, user, tokens, steps,
                          temperature, top_k, seed, gpu),
        Command::Chat { path, system, turns, steps, temperature, top_k, seed } =>
            chat_gemma4_cli(&path, system, turns, steps, temperature, top_k, seed),
        Command::MtpDraft { target, drafter, prompt, system, k } =>
            mtp_draft_cli(&target, &drafter, prompt, system, k),
        Command::MtpGen { target, drafter, prompt, system, k, steps, temperature, seed } =>
            mtp_gen_cli(&target, &drafter, prompt, system, k, steps, temperature, seed),
    }
}

fn generate_text(path: &std::path::Path, prompt_text: Option<String>,
                 system: Option<String>, user: Option<String>,
                 tokens: Option<Vec<u32>>, steps: usize,
                 temperature: f32, top_k: usize, seed: u64, gpu: bool) -> anyhow::Result<()> {
    use reinstinct_engine::sampling::{Rng, sample_temp_topk};
    use reinstinct_engine::tokenizer::Tokenizer;

    let g = GgufFile::open(path)?;
    let arch = g.metadata_get("general.architecture")
        .and_then(|v| v.as_str()).unwrap_or("<unknown>");
    if arch == "gemma4" {
        return generate_text_gemma4(&g, path, prompt_text, system, user, tokens,
                                    steps, temperature, top_k, seed, gpu);
    }
    // Both qwen and gemma4 support --system / --user via their respective
    // chat templates. Other architectures don't have one wired yet.
    let is_qwen = matches!(arch, "qwen35" | "qwen35moe");
    // Typed model — config + quantized tensor refs only. The f32
    // CPU oracle (Qwen35F32Model) is loaded lazily in the CPU branch;
    // building it for --gpu would needlessly materialise the whole
    // model in host RAM (87 GB+ on a 27B model — OOM).
    let model = Qwen35Model::load(&g)?;
    let cfg = &model.config;

    // Prompt resolution:
    //   --system/--user (qwen chat template) > --prompt text > --tokens > [EOS].
    let prompt: Vec<u32> = if system.is_some() || user.is_some() {
        if !is_qwen {
            anyhow::bail!("--system / --user not supported for {arch}; only gemma4 and qwen35.");
        }
        use reinstinct_engine::chat::{ChatMessage, Role, format_qwen3};
        let tok = Tokenizer::from_gguf(&g).map_err(anyhow::Error::msg)?;
        let user_text = user.clone()
            .or_else(|| prompt_text.clone())
            .ok_or_else(|| anyhow::anyhow!(
                "--system was set but no user content (pass --user or --prompt)"))?;
        let mut msgs: Vec<ChatMessage> = Vec::new();
        if let Some(s) = &system {
            msgs.push(ChatMessage { role: Role::System, content: s.clone() });
        }
        msgs.push(ChatMessage { role: Role::User, content: user_text });
        format_qwen3(&tok, &msgs, true).map_err(anyhow::Error::msg)?
    } else if let Some(text) = &prompt_text {
        let tok = Tokenizer::from_gguf(&g).map_err(anyhow::Error::msg)?;
        let ids = tok.encode(text);
        if ids.is_empty() { anyhow::bail!("prompt encoded to zero tokens"); }
        ids
    } else {
        tokens.unwrap_or_else(|| vec![cfg.eos_token_id])
    };

    println!("model       = {}", path.display());
    println!("backend     = {}", if gpu { "GPU (HIP)" } else { "CPU" });
    println!("prompt      = {prompt:?} ({} tokens)", prompt.len());
    println!("steps       = {steps}");
    println!("sampling    = temp={temperature} top_k={top_k} seed={seed}");
    let max_seq = prompt.len() + steps + 4;
    let mut rng = Rng::new(seed);
    let mut all = prompt.clone();

    let t0 = std::time::Instant::now();
    if gpu {
        use reinstinct_engine::hip;
        use reinstinct_engine::runtime::{KernelCache, qwen35::{GpuQwen35, Qwen35GpuState}};
        if hip::device_count().ok().unwrap_or(0) < 1 { anyhow::bail!("no HIP device"); }
        let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
        let cache = KernelCache::new().map_err(anyhow::Error::msg)?;
        let gpu = GpuQwen35::new(&model, &g, &cache, max_seq).map_err(anyhow::Error::msg)?;
        let mut state = Qwen35GpuState::new(&model, max_seq).map_err(anyhow::Error::msg)?;

        // REINSTINCT_PREFILL: batched-prefill benchmark — run the prefill,
        // print timing + top-10, and exit (skips generation). Mirrors the
        // gemma4 path so the harness has one interface for both arches.
        if std::env::var_os("REINSTINCT_PREFILL").is_some() {
            let t = std::time::Instant::now();
            let lg = if prompt.len() > 1 {
                gpu.forward_tokens_batched(&prompt, &mut state).map_err(anyhow::Error::msg)?
            } else {
                gpu.forward_tokens(&prompt, &mut state).map_err(anyhow::Error::msg)?
            };
            let el = t.elapsed().as_secs_f64();
            println!("batched prefill = {:.1} ms  ({} tokens, {:.2} ms/token)",
                     el * 1e3, prompt.len(), el * 1e3 / prompt.len() as f64);
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_unstable_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
            for &t in idx.iter().take(10) {
                println!("  token {t:>8}  logit {:>9.4}", lg[t]);
            }
            return Ok(());
        }

        // Prefill the prompt in one batched pass (rocBLAS GEMM); fall
        // back to the sequential path for a single-token prompt.
        let t_pre = std::time::Instant::now();
        let mut logits = if prompt.len() > 1 {
            gpu.forward_tokens_batched(&prompt, &mut state).map_err(anyhow::Error::msg)?
        } else {
            gpu.forward_tokens(&prompt, &mut state).map_err(anyhow::Error::msg)?
        };
        println!("prefill       = {:.3} s ({} tokens, batched)",
            t_pre.elapsed().as_secs_f32(), prompt.len());
        // Capture the decode forward into a parametric HIP graph — the
        // graph reads `d_pos`, so one capture replays for every step,
        // eliding the per-kernel launch overhead. `REINSTINCT_NO_GRAPH`
        // forces the per-kernel path.
        // REINSTINCT_MOE_PROFILE needs the per-kernel path (its per-stage
        // timer syncs the stream, which a captured graph can't contain).
        let use_graph = std::env::var_os("REINSTINCT_NO_GRAPH").is_none()
                     && std::env::var_os("REINSTINCT_MOE_PROFILE").is_none();
        // Capture only when the graph is used — the profiler's per-stage
        // syncs cannot run inside a stream capture.
        let graph = if use_graph {
            Some(gpu.capture_forward_graph(&mut state).map_err(anyhow::Error::msg)?)
        } else {
            println!("backend mode  = per-kernel");
            None
        };
        // Decode timer — steady-state per-token cost, excluding the
        // one-time weight load and the prompt prefill.
        let t_dec = std::time::Instant::now();
        for _ in 0..steps {
            let tok = sample_temp_topk(&logits, temperature, top_k, &mut rng);
            all.push(tok);
            if tok == cfg.eos_token_id { break; }
            logits = match &graph {
                Some(g) => gpu.forward_token_via_graph(g, tok, &mut state)
                              .map_err(anyhow::Error::msg)?,
                None    => gpu.forward_token(tok, &mut state).map_err(anyhow::Error::msg)?,
            };
        }
        let n_dec = all.len() - prompt.len();
        if n_dec > 0 {
            let d = t_dec.elapsed().as_secs_f64();
            println!("decode        = {:.1} ms/token ({:.1} tok/s) over {n_dec} forwards",
                d * 1e3 / n_dec as f64, n_dec as f64 / d);
        }
        let prof = gpu.moe_prof_report();
        if !prof.is_empty() {
            let tot: f64 = prof.iter().map(|(_, t)| t).sum();
            println!("\n--- MoE decode per-stage ({n_dec} steps, sync-per-lap) ---");
            for (label, ms) in &prof {
                println!("  {label:<14} {ms:8.1} ms  {:5.1}%  ({:.3} ms/step)",
                         100.0 * ms / tot, ms / n_dec.max(1) as f64);
            }
            println!("  {:<14} {tot:8.1} ms", "TOTAL");
        }
    } else {
        // CPU oracle — needs the f32-dequantized weights.
        let m = Qwen35F32Model::load(&g)?;
        let mut state = m.new_state(max_seq);
        let mut logits = m.forward_tokens(&prompt, &mut state);
        for _ in 0..steps {
            let tok = sample_temp_topk(&logits, temperature, top_k, &mut rng);
            all.push(tok);
            if tok == cfg.eos_token_id { break; }
            logits = m.forward_token(tok, &mut state);
        }
    }
    let elapsed = t0.elapsed();
    let new_tokens = all.len() - prompt.len();
    println!("\ngenerated   = {} tokens in {:.2} s ({:.1} tok/s)",
        new_tokens, elapsed.as_secs_f64(), new_tokens as f64 / elapsed.as_secs_f64());
    println!("output ids  = {all:?}");

    // Decode through the GGUF tokenizer if we can.
    match reinstinct_engine::tokenizer::Tokenizer::from_gguf(&g) {
        Ok(tok) => {
            println!("\n--- prompt ---\n{}", tok.decode(&prompt));
            println!("\n--- generated ---\n{}", tok.decode(&all[prompt.len()..]));
            println!("\n--- full output ---\n{}", tok.decode(&all));
        }
        Err(e) => println!("\n(tokenizer decode unavailable: {e})"),
    }
    Ok(())
}

/// Gemma 4 generation via the CPU oracle. Prompt is given as token ids
/// (`--tokens`) — Gemma uses a SentencePiece tokenizer the engine's
/// GPT2-style BPE module doesn't cover yet. Prints token ids + top-K.
fn generate_text_gemma4(g: &GgufFile, path: &std::path::Path,
                        prompt_text: Option<String>,
                        system: Option<String>, user: Option<String>,
                        tokens: Option<Vec<u32>>, steps: usize,
                        temperature: f32, top_k: usize, seed: u64, gpu: bool) -> anyhow::Result<()> {
    use reinstinct_engine::sampling::{Rng, sample_temp_topk};
    use reinstinct_engine::cpu::gemma4::Gemma4CpuModel;
    use reinstinct_engine::model::gemma4::Gemma4Model;
    use reinstinct_engine::tokenizer::GemmaTokenizer;
    use reinstinct_engine::chat::{ChatMessage, Role, format_gemma4};

    let cfg_eos = Gemma4Model::load(g).map_err(anyhow::Error::msg)?.config.eos_token_id;
    // Gemma 4 SentencePiece tokenizer — encodes --prompt text and
    // decodes the generated ids back to text.
    let tok = GemmaTokenizer::from_gguf(g).ok();
    let prompt: Vec<u32> = if system.is_some() || user.is_some() {
        // Chat mode: assemble messages via the Gemma 4 chat template.
        // --user falls back to --prompt's text so users can mix conventions.
        let t = tok.as_ref().ok_or_else(||
            anyhow::anyhow!("--system / --user: gemma4 tokenizer not available"))?;
        let user_text = user.clone()
            .or_else(|| prompt_text.clone())
            .ok_or_else(|| anyhow::anyhow!(
                "--system was set but no user content (pass --user or --prompt)"))?;
        let mut msgs: Vec<ChatMessage> = Vec::new();
        if let Some(s) = &system {
            msgs.push(ChatMessage { role: Role::System, content: s.clone() });
        }
        msgs.push(ChatMessage { role: Role::User, content: user_text });
        format_gemma4(t, &msgs, true).map_err(anyhow::Error::msg)?
    } else if let Some(text) = &prompt_text {
        let t = tok.as_ref().ok_or_else(||
            anyhow::anyhow!("--prompt: this GGUF has no usable gemma4 tokenizer"))?;
        let mut ids = vec![t.bos_id];
        ids.extend(t.encode(text));
        ids
    } else {
        tokens.unwrap_or_else(|| vec![cfg_eos])
    };

    println!("model       = {} (gemma4)", path.display());
    println!("backend     = {}", if gpu { "GPU (HIP)" } else { "CPU oracle" });
    println!("prompt      = {prompt:?} ({} tokens)", prompt.len());
    println!("steps       = {steps}");

    let mut rng = Rng::new(seed);
    let mut all = prompt.clone();
    let t0 = std::time::Instant::now();
    let logits;

    if gpu {
        use reinstinct_engine::hip;
        use reinstinct_engine::runtime::{KernelCache, gemma4::{GpuGemma4, Gemma4GpuState}};
        if hip::device_count().ok().unwrap_or(0) < 1 { anyhow::bail!("no HIP device"); }
        let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
        let cache = KernelCache::new().map_err(anyhow::Error::msg)?;
        let model = Gemma4Model::load(g).map_err(anyhow::Error::msg)?;
        let max_seq = prompt.len() + steps + 8;
        let t_load = std::time::Instant::now();
        let gm = GpuGemma4::new(&model, g, &cache, max_seq).map_err(anyhow::Error::msg)?;
        println!("weights load = {:.2} s", t_load.elapsed().as_secs_f32());

        let mut state = Gemma4GpuState::new(&model, max_seq).map_err(anyhow::Error::msg)?;

        // REINSTINCT_PREFILL: batched-prefill benchmark — run the prefill,
        // print timing + top-10, and exit (skips generation).
        if std::env::var_os("REINSTINCT_PREFILL").is_some() {
            let t = std::time::Instant::now();
            let lg = gm.prefill_forward(&prompt, &mut state)
                .map_err(anyhow::Error::msg)?;
            let el = t.elapsed().as_secs_f64();
            println!("batched prefill = {:.1} ms  ({} tokens, {:.2} ms/token)",
                     el * 1e3, prompt.len(), el * 1e3 / prompt.len() as f64);
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_unstable_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
            for &t in idx.iter().take(10) {
                println!("  token {t:>8}  logit {:>9.4}", lg[t]);
            }
            return Ok(());
        }

        // Capture the decode forward once into a parametric HIP graph —
        // decode then replays it with a single submission per token.
        // REINSTINCT_MOE_PROFILE needs the per-kernel path (its per-stage
        // timer syncs the stream, which a captured graph can't contain).
        let use_graph = std::env::var_os("REINSTINCT_NO_GRAPH").is_none()
                     && std::env::var_os("REINSTINCT_MOE_PROFILE").is_none();
        let t_cap = std::time::Instant::now();
        let graph = if use_graph {
            let g = gm.capture_forward_graph(&state).map_err(anyhow::Error::msg)?;
            println!("graph capture = {:.2} s", t_cap.elapsed().as_secs_f32());
            Some(g)
        } else {
            println!("backend mode  = per-kernel");
            None
        };
        let fwd = |gm: &GpuGemma4, t: u32, st: &mut Gemma4GpuState| {
            match &graph {
                Some(g) => gm.forward_via_graph(g, t, st),
                None    => gm.forward_token(t, st),
            }
        };
        // Prefill the prompt in one batched pass — this populates every
        // layer's KV cache, so decode continues straight from position P.
        let t_prefill = std::time::Instant::now();
        let mut lg = gm.prefill_forward(&prompt, &mut state).map_err(anyhow::Error::msg)?;
        let pf = t_prefill.elapsed().as_secs_f64();
        println!("prefill      = {:.1} ms ({} tokens, {:.2} ms/token)",
                 pf * 1e3, prompt.len(), pf * 1e3 / prompt.len() as f64);
        // Decode timer — generated tokens only.
        let t_decode = std::time::Instant::now();
        for _ in 0..steps {
            let tok = sample_temp_topk(&lg, temperature, top_k, &mut rng);
            all.push(tok);
            if tok == cfg_eos { break; }
            lg = fwd(&gm, tok, &mut state).map_err(anyhow::Error::msg)?;
        }
        let n_gen = all.len() - prompt.len();
        if n_gen > 0 {
            println!("decode       = {:.1} ms/token ({:.1} tok/s) over {n_gen} forwards",
                t_decode.elapsed().as_secs_f64() * 1e3 / n_gen as f64,
                n_gen as f64 / t_decode.elapsed().as_secs_f64());
        }
        let prof = gm.moe_prof_report();
        if !prof.is_empty() {
            let tot: f64 = prof.iter().map(|(_, t)| t).sum();
            println!("\n--- MoE decode per-stage ({n_gen} steps, sync-per-lap) ---");
            for (label, ms) in &prof {
                println!("  {label:<16} {ms:8.1} ms  {:5.1}%  ({:.3} ms/step)",
                         100.0 * ms / tot, ms / n_gen.max(1) as f64);
            }
            println!("  {:<16} {tot:8.1} ms", "TOTAL");
        }
        // One traced forward for a per-block timing breakdown.
        let probe = *all.last().unwrap();
        let (tlg, e_ms, blk_ms, o_ms) =
            gm.forward_token_timed(probe, &mut state).map_err(anyhow::Error::msg)?;
        let total: f32 = e_ms + blk_ms.iter().sum::<f32>() + o_ms;
        let model = Gemma4Model::load(g).map_err(anyhow::Error::msg)?;
        use reinstinct_engine::model::gemma4::AttnKind;
        let (mut sw, mut swn, mut fl, mut fln) = (0.0f32, 0usize, 0.0f32, 0usize);
        for (i, &t) in blk_ms.iter().enumerate() {
            match model.config.attn_kinds[i] {
                AttnKind::Sliding => { sw += t; swn += 1; }
                AttnKind::Full    => { fl += t; fln += 1; }
            }
        }
        println!("\n--- GPU per-stage breakdown (hipEvent ms) ---");
        println!("  total           {total:>8.3} ms");
        println!("  embed           {e_ms:>8.3} ms");
        println!("  blocks sliding  {sw:>8.3} ms  ({swn} blocks, {:.3} ms each)",
                 if swn>0 {sw/swn as f32} else {0.0});
        println!("  blocks full     {fl:>8.3} ms  ({fln} blocks, {:.3} ms each)",
                 if fln>0 {fl/fln as f32} else {0.0});
        println!("  output_proj     {o_ms:>8.3} ms");
        let _ = tlg;
        logits = lg;
    } else {
        let g_owned = GgufFile::open(path)?;
        let m = Gemma4CpuModel::load(g_owned).map_err(anyhow::Error::msg)?;
        let mut state = m.new_state();
        let mut lg = Vec::new();
        for &t in &prompt { lg = m.forward_token(t, &mut state).map_err(anyhow::Error::msg)?; }
        for _ in 0..steps {
            let tok = sample_temp_topk(&lg, temperature, top_k, &mut rng);
            all.push(tok);
            if tok == cfg_eos { break; }
            lg = m.forward_token(tok, &mut state).map_err(anyhow::Error::msg)?;
        }
        logits = lg;
    }

    let elapsed = t0.elapsed();
    let new_tokens = all.len() - prompt.len();
    println!("\ngenerated   = {} tokens in {:.1} s ({:.3} s/token)",
        new_tokens, elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / (prompt.len() + new_tokens).max(1) as f64);
    println!("output ids  = {all:?}");
    if let Some(t) = &tok {
        println!("output text = {:?}", t.decode(&all));
    }

    // Top-K of the final logits for an architecture sanity check.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    println!("\n--- top {} logits (final position) ---", top_k.min(10));
    for &i in idx.iter().take(top_k.min(10)) {
        println!("  token {i:>7}  logit {:>9.4}", logits[i]);
    }
    let (mn, mx) = logits.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    let nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
    println!("logit range = [{mn:.3}, {mx:.3}], nonfinite = {nonfinite}");
    Ok(())
}

fn gpu_bench(path: &std::path::Path, iters: usize, token: Option<u32>) -> anyhow::Result<()> {
    use reinstinct_engine::hip;
    use reinstinct_engine::runtime::{KernelCache, qwen35::{GpuQwen35, Qwen35GpuState}};
    use reinstinct_engine::model::qwen3_5::BlockKind;

    let n = hip::device_count().map_err(anyhow::Error::msg)?;
    if n == 0 { anyhow::bail!("no HIP device"); }
    let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
    let cache = KernelCache::new().map_err(anyhow::Error::msg)?;

    let g = GgufFile::open(path)?;
    // Profiling only needs the config + GPU-resident weights — load the
    // lightweight typed model, not the f32 oracle (which would OOM the
    // host on the 27B).
    let m = Qwen35Model::load(&g)?;
    let cfg = &m.config;
    let token = token.unwrap_or(cfg.eos_token_id);

    println!("model = {}", path.display());
    println!("token = {token}, iterations = {iters}");
    println!("loading weights to device...");
    let t0 = std::time::Instant::now();
    let gpu = GpuQwen35::new(&m, &g, &cache, iters + 4).map_err(anyhow::Error::msg)?;
    println!("  load took {:.2} s", t0.elapsed().as_secs_f64());

    let mut state = Qwen35GpuState::new(&m, iters + 4).map_err(anyhow::Error::msg)?;

    // Warm up once (compiles + caches paged-in, etc.)
    let _ = gpu.forward_token(token, &mut state).map_err(anyhow::Error::msg)?;
    state.reset().map_err(anyhow::Error::msg)?;

    let mut times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let _ = gpu.forward_token(token, &mut state).map_err(anyhow::Error::msg)?;
        times_us.push(t.elapsed().as_micros() as u64);
        state.reset().map_err(anyhow::Error::msg)?;
    }
    times_us.sort_unstable();
    let median  = times_us[times_us.len() / 2] as f64 / 1000.0;
    let mean    = times_us.iter().sum::<u64>() as f64 / times_us.len() as f64 / 1000.0;
    let min     = times_us[0] as f64 / 1000.0;
    let max     = *times_us.last().unwrap() as f64 / 1000.0;
    println!("\n--- GPU forward_token (direct), {iters} iterations ---");
    println!("  median  {median:>8.3} ms  ({:>5.1} tok/s)", 1000.0 / median);
    println!("  mean    {mean:>8.3} ms");
    println!("  min     {min:>8.3} ms");
    println!("  max     {max:>8.3} ms");

    // Per-stage breakdown via HIP events. Run a few traced iterations
    // and average to smooth out single-shot noise; report the per-stage
    // and per-block-kind contributions.
    state.reset().map_err(anyhow::Error::msg)?;
    let trace_iters = 5usize;
    let mut sum_embed = 0.0_f32;
    let mut sum_norm  = 0.0_f32;
    let mut sum_proj  = 0.0_f32;
    let mut sum_block = vec![0.0_f32; m.block_kinds.len()];
    let mut sum_total = 0.0_f32;
    for _ in 0..trace_iters {
        let (_logits, t) = gpu.forward_token_traced(token, &mut state).map_err(anyhow::Error::msg)?;
        sum_embed += t.embed_ms;
        sum_norm  += t.output_norm_ms;
        sum_proj  += t.output_proj_ms;
        for (acc, v) in sum_block.iter_mut().zip(t.block_ms.iter()) { *acc += *v; }
        sum_total += t.total_ms;
        state.reset().map_err(anyhow::Error::msg)?;
    }
    let n = trace_iters as f32;
    let total = sum_total / n;
    let embed = sum_embed / n;
    let norm  = sum_norm / n;
    let proj  = sum_proj / n;
    let mut sum_lin  = 0.0_f32; let mut count_lin  = 0usize;
    let mut sum_full = 0.0_f32; let mut count_full = 0usize;
    for (acc, &kind) in sum_block.iter().zip(m.block_kinds.iter()) {
        let avg = acc / n;
        match kind {
            BlockKind::LinearAttention => { sum_lin += avg; count_lin += 1; }
            BlockKind::FullAttention   => { sum_full += avg; count_full += 1; }
        }
    }
    let pct = |x: f32| 100.0 * x / total;
    println!("\n--- per-stage GPU breakdown ({} traced iters, hipEvent ms) ---", trace_iters);
    println!("  total           {total:>8.3} ms  (event sum)");
    println!("  embed           {embed:>8.3} ms ({:>4.1}%)", pct(embed));
    println!("  blocks (linear) {sum_lin:>8.3} ms ({:>4.1}%)  -- {count_lin} blocks, {:.3} ms each",
        pct(sum_lin), if count_lin > 0 { sum_lin / count_lin as f32 } else { 0.0 });
    println!("  blocks (full)   {sum_full:>8.3} ms ({:>4.1}%)  -- {count_full} blocks, {:.3} ms each",
        pct(sum_full), if count_full > 0 { sum_full / count_full as f32 } else { 0.0 });
    println!("  output_norm     {norm:>8.3} ms ({:>4.1}%)", pct(norm));
    println!("  output_proj     {proj:>8.3} ms ({:>4.1}%)", pct(proj));

    // Per-kernel breakdown for one GDN block (block 0). Pick an L block.
    let lin_idx = m.block_kinds.iter()
        .position(|k| matches!(k, BlockKind::LinearAttention))
        .ok_or_else(|| anyhow::anyhow!("no L block"))?;
    state.reset().map_err(anyhow::Error::msg)?;
    let trace_iters_gdn = 5usize;
    let mut sum_kernels: std::collections::BTreeMap<&'static str, f32> =
        std::collections::BTreeMap::new();
    let mut order: Vec<&'static str> = Vec::new();
    for it in 0..trace_iters_gdn {
        let (_logits, ks) = gpu.forward_token_traced_gdn(token, &mut state, lin_idx)
            .map_err(anyhow::Error::msg)?;
        if it == 0 { for (n, _) in &ks { order.push(n); } }
        for (n, ms) in ks { *sum_kernels.entry(n).or_insert(0.0) += ms; }
        state.reset().map_err(anyhow::Error::msg)?;
    }
    let total_gdn: f32 = sum_kernels.values().sum::<f32>() / trace_iters_gdn as f32;
    println!("\n--- one GDN block kernel breakdown ({} iters, ms each) ---", trace_iters_gdn);
    println!("  block index = {lin_idx} (L)");
    for n in &order {
        let avg = sum_kernels[n] / trace_iters_gdn as f32;
        println!("  {n:<22} {avg:>7.4} ms ({:>4.1}%)", 100.0 * avg / total_gdn);
    }
    println!("  {:<22} {total_gdn:>7.4} ms (sum of GDN kernels in one block)", "TOTAL");

    // HIP graph capture: capture the full forward chain once at pos=0
    // for this token, then time hipGraphLaunch + sync + D2H per call.
    // The bench resets state between iters so capturing at pos=0 is
    // valid for every iteration here.
    state.reset().map_err(anyhow::Error::msg)?;
    let t_cap = std::time::Instant::now();
    let exec = gpu.capture_forward_graph(&mut state).map_err(anyhow::Error::msg)?;
    println!("\nHIP graph capture + instantiate took {:.3} ms",
        t_cap.elapsed().as_secs_f64() * 1000.0);
    state.reset().map_err(anyhow::Error::msg)?;
    let _ = gpu.forward_token_via_graph(&exec, token, &mut state).map_err(anyhow::Error::msg)?;  // warmup
    state.reset().map_err(anyhow::Error::msg)?;

    let mut g_times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let _ = gpu.forward_token_via_graph(&exec, token, &mut state).map_err(anyhow::Error::msg)?;
        g_times_us.push(t.elapsed().as_micros() as u64);
        state.reset().map_err(anyhow::Error::msg)?;
    }
    g_times_us.sort_unstable();
    let g_median = g_times_us[g_times_us.len() / 2] as f64 / 1000.0;
    let g_mean   = g_times_us.iter().sum::<u64>() as f64 / g_times_us.len() as f64 / 1000.0;
    let g_min    = g_times_us[0] as f64 / 1000.0;
    let g_max    = *g_times_us.last().unwrap() as f64 / 1000.0;
    println!("\n--- GPU forward_token (HIP graph), {iters} iterations ---");
    println!("  median  {g_median:>8.3} ms  ({:>5.1} tok/s)", 1000.0 / g_median);
    println!("  mean    {g_mean:>8.3} ms");
    println!("  min     {g_min:>8.3} ms");
    println!("  max     {g_max:>8.3} ms");
    let graph_speedup = median / g_median;
    let label = if graph_speedup >= 1.0 { "speedup over direct" } else { "slowdown vs direct" };
    println!("  graph: {graph_speedup:.2}× {label}");

    // CPU baseline — only when the f32 oracle fits in host RAM. The
    // f32 model is ≈7× the GGUF file; skip it for large models so the
    // GPU profile still runs (the 27B f32 oracle would be ~115 GB).
    let gguf_bytes = std::fs::metadata(path).map(|md| md.len()).unwrap_or(0);
    if gguf_bytes < 4 * 1024 * 1024 * 1024 {
        println!("\n--- CPU forward_token, {iters} iterations ---");
        let cpu_m = Qwen35F32Model::load(&g)?;
        let mut cpu_state = cpu_m.new_state(iters + 4);
        let _ = cpu_m.forward_token(token, &mut cpu_state);  // warmup
        cpu_state.reset();
        let mut cpu_times_us = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let _ = cpu_m.forward_token(token, &mut cpu_state);
            cpu_times_us.push(t.elapsed().as_micros() as u64);
            cpu_state.reset();
        }
        cpu_times_us.sort_unstable();
        let cpu_median = cpu_times_us[cpu_times_us.len() / 2] as f64 / 1000.0;
        println!("  median  {cpu_median:>8.3} ms  ({:>5.1} tok/s)", 1000.0 / cpu_median);
        let speedup = cpu_median / median;
        let label = if speedup >= 1.0 { "speedup" } else { "slowdown" };
        println!("\nGPU vs CPU: {speedup:.2}× {label} (median)");
    } else {
        println!("\n(CPU baseline skipped — f32 oracle too large for host RAM)");
    }
    Ok(())
}

fn hip_info(mb: usize, iters: usize) -> anyhow::Result<()> {
    use reinstinct_engine::hip;
    let n = hip::device_count().map_err(anyhow::Error::msg)?;
    println!("HIP devices = {n}");
    if n == 0 { return Ok(()); }

    for d in 0..n {
        let name  = hip::device_name(d).map_err(anyhow::Error::msg)?;
        let total = hip::device_total_mem(d).map_err(anyhow::Error::msg)?;
        println!("  [{d}] {name}  total VRAM = {:.2} GB", total as f64 / (1u64 << 30) as f64);
    }

    let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
    let (free, total) = hip::mem_info().map_err(anyhow::Error::msg)?;
    println!("\ndevice 0: {:.2} / {:.2} GB free",
        free as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64);

    let n_elems = (mb * (1 << 20)) / std::mem::size_of::<f32>();
    let host: Vec<f32> = (0..n_elems).map(|i| (i as f32) * 1.0e-3).collect();
    let mut back = vec![0.0f32; n_elems];
    println!("\nbandwidth probe: {} MB per direction, {} iters", mb, iters);

    let buf = hip::DeviceBuf::from_slice(&host).map_err(anyhow::Error::msg)?;
    // Verify correctness on first round.
    buf.copy_to_host(&mut back).map_err(anyhow::Error::msg)?;
    for i in 0..n_elems {
        if host[i].to_bits() != back[i].to_bits() {
            anyhow::bail!("round-trip mismatch at {i}");
        }
    }

    let bytes = (mb << 20) as f64;
    let mut h2d_total = 0.0_f64;
    let mut d2h_total = 0.0_f64;
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        buf.copy_from_host(&host).map_err(anyhow::Error::msg)?;
        h2d_total += t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        buf.copy_to_host(&mut back).map_err(anyhow::Error::msg)?;
        d2h_total += t1.elapsed().as_secs_f64();
    }
    let h2d = bytes / (h2d_total / iters as f64) / 1e9;
    let d2h = bytes / (d2h_total / iters as f64) / 1e9;
    println!("  H2D  {h2d:>6.2} GB/s   ({:.3} ms / copy)", 1000.0 * h2d_total / iters as f64);
    println!("  D2H  {d2h:>6.2} GB/s   ({:.3} ms / copy)", 1000.0 * d2h_total / iters as f64);

    // D2D streaming bandwidth: an in-VRAM copy moves `bytes` read + `bytes`
    // written, so effective HBM traffic per copy is 2*bytes. This is the
    // achievable streaming-bandwidth ceiling to compare matvec GB/s against.
    let dst: hip::DeviceBuf<f32> = hip::DeviceBuf::new(n_elems).map_err(anyhow::Error::msg)?;
    dst.copy_from_device_at(&buf, 0).map_err(anyhow::Error::msg)?; // warm up
    _dev.synchronize().map_err(anyhow::Error::msg)?;
    let mut d2d_total = 0.0_f64;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        dst.copy_from_device_at(&buf, 0).map_err(anyhow::Error::msg)?;
        _dev.synchronize().map_err(anyhow::Error::msg)?;
        d2d_total += t.elapsed().as_secs_f64();
    }
    let d2d = 2.0 * bytes / (d2d_total / iters as f64) / 1e9;
    println!("  D2D  {d2d:>6.2} GB/s   ({:.3} ms / copy, read+write traffic)",
        1000.0 * d2d_total / iters as f64);

    // Compute-kernel streaming read: the bandwidth a *compute kernel* (not
    // the DMA copy engine) sustains on a perfectly-coalesced contiguous
    // float4 read. This is the honest ceiling a matvec kernel can target.
    const STREAM_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__
void stream_read(const float4* __restrict__ in, float* __restrict__ out,
                 unsigned int n4) {
    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    float sx = 0.f, sy = 0.f, sz = 0.f, sw = 0.f;
    for (unsigned int i = tid; i < n4; i += stride) {
        float4 v = in[i];
        sx += v.x; sy += v.y; sz += v.z; sw += v.w;
    }
    out[tid] = sx + sy + sz + sw;
}
"#;
    let cache = reinstinct_engine::runtime::KernelCache::new().map_err(anyhow::Error::msg)?;
    let hsaco = cache.compile("stream_read", STREAM_SRC).map_err(anyhow::Error::msg)?;
    let module = hip::Module::load(&hsaco).map_err(anyhow::Error::msg)?;
    let f = module.function("stream_read").map_err(anyhow::Error::msg)?;
    let stream = hip::Stream::new().map_err(anyhow::Error::msg)?;
    let (grid, block) = (480u32, 256u32);
    let sink: hip::DeviceBuf<f32> = hip::DeviceBuf::new((grid * block) as usize)
        .map_err(anyhow::Error::msg)?;
    let n4 = (n_elems / 4) as u32;
    let launch = |st: &hip::Stream| -> Result<(), String> {
        let mut ia = buf.raw_ptr();
        let mut oa = sink.raw_ptr();
        let mut na = n4;
        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut ia as *mut _ as *mut std::ffi::c_void, &mut oa as *mut _ as *mut std::ffi::c_void,
            &mut na as *mut _ as *mut std::ffi::c_void];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(st), &mut args) }
    };
    launch(&stream).map_err(anyhow::Error::msg)?; // warm up
    _dev.synchronize().map_err(anyhow::Error::msg)?;
    let start = hip::Event::new().map_err(anyhow::Error::msg)?;
    let stop  = hip::Event::new().map_err(anyhow::Error::msg)?;
    start.record(&stream).map_err(anyhow::Error::msg)?;
    for _ in 0..iters { launch(&stream).map_err(anyhow::Error::msg)?; }
    stop.record(&stream).map_err(anyhow::Error::msg)?;
    _dev.synchronize().map_err(anyhow::Error::msg)?;
    let ms = hip::Event::elapsed_time(&start, &stop).map_err(anyhow::Error::msg)? as f64;
    let read = bytes / (ms / 1000.0 / iters as f64) / 1e9;
    println!("  kernel-read  {read:>6.2} GB/s   ({:.3} ms / pass, compute-kernel ceiling)",
        ms / iters as f64);

    // Per-kernel dispatch cost — the fixed overhead of a kernel that does
    // ~no work, measured both as a HIP-graph replay (the path decode
    // uses: CPU dispatch amortised, leaves GPU-side per-dispatch cost)
    // and as direct launches (adds CPU dispatch). Multiply by the kernel
    // count to size what kernel fusion can recover.
    const NOOP_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void noop_kernel(int* p) {
    if (threadIdx.x == 0) p[0] += 1;
}
"#;
    let nmod = hip::Module::load(&cache.compile("noop_kernel", NOOP_SRC).map_err(anyhow::Error::msg)?)
        .map_err(anyhow::Error::msg)?;
    let nf = nmod.function("noop_kernel").map_err(anyhow::Error::msg)?;
    let nbuf: hip::DeviceBuf<i32> = hip::DeviceBuf::new(1).map_err(anyhow::Error::msg)?;
    let n_kern = 1024usize;
    let noop = |st: &hip::Stream| -> Result<(), String> {
        let mut p = nbuf.raw_ptr();
        let mut a: [*mut std::ffi::c_void; 1] = [&mut p as *mut _ as *mut std::ffi::c_void];
        unsafe { nf.launch((1, 1, 1), (64, 1, 1), 0, Some(st), &mut a) }
    };

    // Graph replay: capture n_kern noop launches, replay `iters` times.
    use reinstinct_engine::hip::sys::HipStreamCaptureMode;
    hip::Graph::begin_capture(&stream, HipStreamCaptureMode::Global).map_err(anyhow::Error::msg)?;
    for _ in 0..n_kern { noop(&stream).map_err(anyhow::Error::msg)?; }
    let g = hip::Graph::end_capture(&stream).map_err(anyhow::Error::msg)?;
    let gexec = g.instantiate().map_err(anyhow::Error::msg)?;
    gexec.launch(&stream).map_err(anyhow::Error::msg)?;        // warm up
    _dev.synchronize().map_err(anyhow::Error::msg)?;
    let gs = hip::Event::new().map_err(anyhow::Error::msg)?;
    let ge = hip::Event::new().map_err(anyhow::Error::msg)?;
    gs.record(&stream).map_err(anyhow::Error::msg)?;
    for _ in 0..iters { gexec.launch(&stream).map_err(anyhow::Error::msg)?; }
    ge.record(&stream).map_err(anyhow::Error::msg)?;
    _dev.synchronize().map_err(anyhow::Error::msg)?;
    let g_ms = hip::Event::elapsed_time(&gs, &ge).map_err(anyhow::Error::msg)? as f64;
    let g_per = g_ms * 1000.0 / (iters as f64 * n_kern as f64);   // µs / kernel

    // Direct launches: n_kern * iters, wall-clock timed.
    noop(&stream).map_err(anyhow::Error::msg)?;
    _dev.synchronize().map_err(anyhow::Error::msg)?;
    let t = std::time::Instant::now();
    for _ in 0..iters { for _ in 0..n_kern { noop(&stream).map_err(anyhow::Error::msg)?; } }
    _dev.synchronize().map_err(anyhow::Error::msg)?;
    let d_per = t.elapsed().as_secs_f64() * 1e6 / (iters as f64 * n_kern as f64);

    println!("\nper-kernel dispatch cost ({} kernels, {} iters):", n_kern, iters);
    println!("  graph replay  {g_per:.3} µs / kernel   (GPU-side dispatch)");
    println!("  direct launch {d_per:.3} µs / kernel   (+ CPU dispatch)");
    println!("  → 1260 kernels/token ≈ {:.2} ms graph-side", g_per * 1260.0 / 1000.0);
    Ok(())
}

fn bench(path: &std::path::Path, iters: usize, token: Option<u32>) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    let m = Qwen35F32Model::load(&g)?;
    let cfg = &m.model.config;
    let token = token.unwrap_or(cfg.eos_token_id);
    println!("model = {}", path.display());
    println!("token = {token}, iterations = {iters}");

    // Warm up once so loader / page cache effects don't dominate iter[0].
    let mut state = m.new_state(iters + 4);
    let _ = m.forward_token(token, &mut state);

    let mut traces: Vec<ForwardTrace> = Vec::with_capacity(iters);
    state.reset();
    for _ in 0..iters {
        let mut t = ForwardTrace::default();
        let _ = m.forward_token_traced(token, &mut state, Some(&mut t));
        traces.push(t);
        state.reset();
    }

    // Aggregate.
    let n_blocks = m.model.block_kinds.len();
    let mut sum_embed = 0u64;
    let mut sum_norm = 0u64;
    let mut sum_proj = 0u64;
    let mut sum_blocks_lin = 0u64;  // total ns over all linear-attn blocks
    let mut sum_blocks_full = 0u64;
    let mut count_lin = 0usize;
    let mut count_full = 0usize;
    let mut sum_per_block = vec![0u64; n_blocks];
    for t in &traces {
        sum_embed += t.embed_ns;
        sum_norm += t.output_norm_ns;
        sum_proj += t.output_proj_ns;
        for (i, &b) in t.block_ns.iter().enumerate() {
            sum_per_block[i] += b;
            match m.model.block_kinds[i] {
                BlockKind::LinearAttention => { sum_blocks_lin += b; count_lin += 1; }
                BlockKind::FullAttention   => { sum_blocks_full += b; count_full += 1; }
            }
        }
    }
    let n = iters as u64;
    let total: u64 = traces.iter().map(|t| t.total_ns()).sum();
    let avg = total / n;
    let pct = |x: u64| 100.0 * (x as f64) / (total as f64);

    println!("\n--- per-iteration averages ---");
    println!("  total           {:>9.3} ms", (avg as f64) / 1.0e6);
    println!("  embed lookup    {:>9.3} ms ({:>4.1}%)",
        (sum_embed / n) as f64 / 1.0e6, pct(sum_embed));
    println!("  blocks (linear) {:>9.3} ms ({:>4.1}%)  -- {} blocks, {:.3} ms each",
        (sum_blocks_lin / n) as f64 / 1.0e6, pct(sum_blocks_lin),
        count_lin / iters,
        if count_lin > 0 { (sum_blocks_lin as f64) / (count_lin as f64) / 1.0e6 } else { 0.0 });
    println!("  blocks (full)   {:>9.3} ms ({:>4.1}%)  -- {} blocks, {:.3} ms each",
        (sum_blocks_full / n) as f64 / 1.0e6, pct(sum_blocks_full),
        count_full / iters,
        if count_full > 0 { (sum_blocks_full as f64) / (count_full as f64) / 1.0e6 } else { 0.0 });
    println!("  output_norm     {:>9.3} ms ({:>4.1}%)",
        (sum_norm / n) as f64 / 1.0e6, pct(sum_norm));
    println!("  output_proj     {:>9.3} ms ({:>4.1}%)",
        (sum_proj / n) as f64 / 1.0e6, pct(sum_proj));

    println!("\n--- per-block averages (ms) ---");
    for (i, &s) in sum_per_block.iter().enumerate() {
        let kind = match m.model.block_kinds[i] {
            BlockKind::LinearAttention => "L",
            BlockKind::FullAttention   => "F",
        };
        println!("  block {i:>2} {kind}  {:>7.3}", (s / n) as f64 / 1.0e6);
    }
    Ok(())
}

fn debug_embed(path: &std::path::Path, tokens: &[u32]) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    let m = Qwen35F32Model::load(&g)?;
    let h = m.model.config.hidden_size as usize;
    println!("hidden_size = {h}, vocab = {}", m.model.config.vocab_size);
    for &tok in tokens {
        let off = tok as usize * h;
        let row = &m.weights.token_embd[off..off + h];
        let rms: f32 = (row.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
        let max = row.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let head: Vec<f32> = row[..6].to_vec();
        let tail: Vec<f32> = row[h - 6..].to_vec();
        println!("\ntoken {tok:>6}:");
        println!("  rms        = {rms:.6}");
        println!("  max|x|     = {max:.6}");
        println!("  first 6    = {head:?}");
        println!("  last 6     = {tail:?}");
    }
    Ok(())
}

fn generate(path: &std::path::Path, token: Option<u32>, tokens: Option<Vec<u32>>, k: usize, gpu: bool) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    let m = Qwen35F32Model::load(&g)?;
    let cfg = &m.model.config;
    let prompt: Vec<u32> = tokens.unwrap_or_else(|| vec![token.unwrap_or(cfg.eos_token_id)]);
    println!("model         = {}", path.display());
    println!("vocab         = {}", cfg.vocab_size);
    println!("backend       = {}", if gpu { "GPU (HIP)" } else { "CPU" });
    println!("input tokens  = {prompt:?}");

    let logits: Vec<f32> = if gpu {
        use reinstinct_engine::hip;
        use reinstinct_engine::runtime::{KernelCache, qwen35::{GpuQwen35, Qwen35GpuState}};
        let n = hip::device_count().map_err(anyhow::Error::msg)?;
        if n == 0 { anyhow::bail!("no HIP device"); }
        let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
        let cache = KernelCache::new().map_err(anyhow::Error::msg)?;
        let max_seq = prompt.len() + 8;
        let t_load = std::time::Instant::now();
        let gpu = GpuQwen35::new(&m.model, &g, &cache, max_seq).map_err(anyhow::Error::msg)?;
        println!("weights load  = {:.2} s", t_load.elapsed().as_secs_f32());
        let mut state = Qwen35GpuState::new(&m.model,max_seq).map_err(anyhow::Error::msg)?;
        let t0 = std::time::Instant::now();
        let l = gpu.forward_tokens(&prompt, &mut state).map_err(anyhow::Error::msg)?;
        println!("forward took  = {:.3} s ({} tokens, {:.1} ms/token)",
            t0.elapsed().as_secs_f32(), prompt.len(),
            t0.elapsed().as_secs_f64() * 1000.0 / prompt.len() as f64);
        l
    } else {
        let mut state = m.new_state(prompt.len() + 8);
        let t0 = std::time::Instant::now();
        let l = m.forward_tokens(&prompt, &mut state);
        println!("forward took  = {:.3} s ({} tokens, {:.1} ms/token)",
            t0.elapsed().as_secs_f32(), prompt.len(),
            t0.elapsed().as_secs_f64() * 1000.0 / prompt.len() as f64);
        l
    };

    // Compute softmax probability for the top-k for context.
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_logit = indexed[0].1;
    let mut sum_exp = 0.0_f64;
    for &(_, v) in &logits.iter().enumerate().map(|(i, x)| (i, *x)).collect::<Vec<_>>() {
        sum_exp += ((v - max_logit) as f64).exp();
    }

    println!("\n--- top {k} logits ---");
    for &(i, v) in indexed.iter().take(k) {
        let p = ((v - max_logit) as f64).exp() / sum_exp;
        println!("  token {i:>6}  logit {v:>9.4}  p = {p:.4}");
    }

    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &v in &logits {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
    }
    let n = logits.len() as f64;
    let mean = (sum / n) as f32;
    let std = ((sum_sq / n) - (mean as f64).powi(2)).sqrt() as f32;
    println!("\nlogit stats   = min {min:.4}  max {max:.4}  mean {mean:.4}  std {std:.4}");
    Ok(())
}

/// Multi-turn Gemma 4 chat with KV-cache prefix reuse. Prefills the
/// system message once, snapshots the state, and reuses that snapshot
/// for every turn — TTFT for turn N+ is bounded by the per-turn token
/// count, not the system+history length.
fn chat_gemma4_cli(path: &std::path::Path, system: Option<String>,
                   turns: Vec<String>, steps: usize,
                   temperature: f32, top_k: usize, seed: u64) -> anyhow::Result<()> {
    use reinstinct_engine::chat::{ChatMessage, Role, format_gemma4, format_gemma4_user_turn};
    use reinstinct_engine::sampling::{Rng, sample_temp_topk};
    use reinstinct_engine::tokenizer::GemmaTokenizer;
    use reinstinct_engine::model::gemma4::Gemma4Model;
    use reinstinct_engine::runtime::{KernelCache, gemma4::{GpuGemma4, Gemma4GpuState}};
    use reinstinct_engine::hip;

    if turns.is_empty() {
        anyhow::bail!("chat: need at least one --turn");
    }
    let g = GgufFile::open(path)?;
    let arch = g.metadata_get("general.architecture").and_then(|v| v.as_str()).unwrap_or("?");
    if arch != "gemma4" {
        anyhow::bail!("chat is currently gemma4-only (this is {arch})");
    }
    let tok = GemmaTokenizer::from_gguf(&g).map_err(anyhow::Error::msg)?;

    // System prefix — rendered with add_generation_prompt = false so
    // it ends at <turn|>\n, the natural place to splice each turn in.
    let mut prefix_msgs: Vec<ChatMessage> = Vec::new();
    if let Some(s) = &system {
        prefix_msgs.push(ChatMessage { role: Role::System, content: s.clone() });
    }
    let prefix_tokens: Vec<u32> = if prefix_msgs.is_empty() {
        vec![tok.bos_id]
    } else {
        format_gemma4(&tok, &prefix_msgs, false).map_err(anyhow::Error::msg)?
    };

    // Conservative max_seq: prefix + every turn (with its model header
    // and decoded response) all coresident — `chat` never compacts.
    let per_turn_cap = 64 + steps + 16;
    let max_seq = prefix_tokens.len() + turns.len() * per_turn_cap + 32;

    println!("model       = {} (gemma4)", path.display());
    println!("backend     = GPU (HIP)");
    println!("system tok  = {} (prefix prefilled + snapshotted once)", prefix_tokens.len());
    println!("turns       = {}, steps/turn = {steps}", turns.len());

    if hip::device_count().ok().unwrap_or(0) < 1 { anyhow::bail!("no HIP device"); }
    let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
    let cache = KernelCache::new().map_err(anyhow::Error::msg)?;
    let model = Gemma4Model::load(&g).map_err(anyhow::Error::msg)?;
    let cfg_eos = model.config.eos_token_id;
    let gm = GpuGemma4::new(&model, &g, &cache, max_seq).map_err(anyhow::Error::msg)?;
    let mut state = Gemma4GpuState::new(&model, max_seq).map_err(anyhow::Error::msg)?;
    let exec = gm.capture_forward_graph(&state).map_err(anyhow::Error::msg)?;

    // 1) Batched prefill of the system prefix → snapshot.
    let t = std::time::Instant::now();
    state.reset();
    let _ = gm.prefill_forward(&prefix_tokens, &mut state).map_err(anyhow::Error::msg)?;
    let t_prefix = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let snap = state.snapshot().map_err(anyhow::Error::msg)?;
    let t_snap = t.elapsed().as_secs_f64();
    println!("prefix prefill = {:.1} ms ({} tok, snapshot {:.1} ms)",
             t_prefix * 1e3, prefix_tokens.len(), t_snap * 1e3);

    let mut rng = Rng::new(seed);
    for (i, turn_text) in turns.iter().enumerate() {
        // 2) Restore the cached prefix.
        let t_r = std::time::Instant::now();
        state.restore(&snap).map_err(anyhow::Error::msg)?;
        let t_restore = t_r.elapsed().as_secs_f64();
        // 3) Sequential prefill of the per-turn tokens (small — typically
        // tens of tokens; sequential here is simpler than extending
        // prefill_forward to start at a non-zero position).
        let turn_tokens = format_gemma4_user_turn(&tok, turn_text).map_err(anyhow::Error::msg)?;
        let t_t = std::time::Instant::now();
        let mut logits: Vec<f32> = Vec::new();
        for &tk in &turn_tokens {
            logits = gm.forward_via_graph(&exec, tk, &mut state).map_err(anyhow::Error::msg)?;
        }
        let t_turn = t_t.elapsed().as_secs_f64();
        let ttft_ms = (t_restore + t_turn) * 1e3;

        // 4) Decode.
        let t_d = std::time::Instant::now();
        let mut out_ids: Vec<u32> = Vec::with_capacity(steps);
        for _ in 0..steps {
            let tk = sample_temp_topk(&logits, temperature, top_k, &mut rng);
            out_ids.push(tk);
            if tk == cfg_eos { break; }
            logits = gm.forward_via_graph(&exec, tk, &mut state).map_err(anyhow::Error::msg)?;
        }
        let t_decode = t_d.elapsed().as_secs_f64();
        let dec_tps = out_ids.len() as f64 / t_decode;

        let response = tok.decode(&out_ids);
        println!();
        println!("--- turn {} ({} user tok) ---", i + 1, turn_tokens.len());
        println!("  ttft        = {:.1} ms  (restore {:.1} + prefill {:.1})",
                 ttft_ms, t_restore * 1e3, t_turn * 1e3);
        println!("  decode      = {:.1} tok/s over {} tokens", dec_tps, out_ids.len());
        println!("  user        > {}", turn_text);
        println!("  assistant   > {}", response.trim());
    }

    Ok(())
}

/// Spec-decode smoke test for the Gemma 4 MTP drafter. Loads target +
/// drafter, prefills the prompt on the target (which establishes h_prev
/// and populates the shared KV the drafter will attend), then runs the
/// drafter K times printing each proposed token plus the target's own
/// argmax for comparison.
fn mtp_draft_cli(target_path: &std::path::Path, drafter_path: &std::path::Path,
                 prompt_text: Option<String>, system: Option<String>, k: usize)
    -> anyhow::Result<()>
{
    use reinstinct_engine::chat::{ChatMessage, Role, format_gemma4};
    use reinstinct_engine::hip;
    use reinstinct_engine::model::gemma4::Gemma4Model;
    use reinstinct_engine::model::gemma4_assistant::Gemma4AssistantModel;
    use reinstinct_engine::runtime::{KernelCache, gemma4::{GpuGemma4, Gemma4GpuState}};
    use reinstinct_engine::runtime::gemma4_assistant::GpuGemma4Assistant;
    use reinstinct_engine::tokenizer::GemmaTokenizer;

    let target_gguf  = GgufFile::open(target_path)?;
    let drafter_gguf = GgufFile::open(drafter_path)?;
    let tok = GemmaTokenizer::from_gguf(&target_gguf).map_err(anyhow::Error::msg)?;

    // Render the prompt — chat-template path if --system was given.
    let prompt: Vec<u32> = if let Some(s) = &system {
        let user = prompt_text.clone().unwrap_or_default();
        let msgs = vec![
            ChatMessage { role: Role::System, content: s.clone() },
            ChatMessage { role: Role::User,   content: user },
        ];
        format_gemma4(&tok, &msgs, true).map_err(anyhow::Error::msg)?
    } else if let Some(t) = &prompt_text {
        let mut ids = vec![tok.bos_id];
        ids.extend(tok.encode(t));
        ids
    } else {
        anyhow::bail!("mtp-draft: pass --prompt or --system/--prompt");
    };

    println!("target   = {}", target_path.display());
    println!("drafter  = {}", drafter_path.display());
    println!("prompt   = {} tokens", prompt.len());

    if hip::device_count().ok().unwrap_or(0) < 1 { anyhow::bail!("no HIP device"); }
    let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
    let cache = KernelCache::new().map_err(anyhow::Error::msg)?;

    let target_model = Gemma4Model::load(&target_gguf).map_err(anyhow::Error::msg)?;
    let max_seq = prompt.len() + k + 16;
    let t = std::time::Instant::now();
    let gm = GpuGemma4::new(&target_model, &target_gguf, &cache, max_seq)
        .map_err(anyhow::Error::msg)?;
    println!("target loaded in {:.2} s", t.elapsed().as_secs_f32());

    let drafter_model = Gemma4AssistantModel::load(&drafter_gguf).map_err(anyhow::Error::msg)?;
    let t = std::time::Instant::now();
    let drafter = GpuGemma4Assistant::new(&drafter_model, &drafter_gguf, &gm, &cache)
        .map_err(anyhow::Error::msg)?;
    println!("drafter loaded in {:.2} s", t.elapsed().as_secs_f32());

    let mut state = Gemma4GpuState::new(&target_model, max_seq).map_err(anyhow::Error::msg)?;
    state.reset();

    // Prefill the prompt on the target (P-1 positions populate the
    // shared KV; the final token's forward writes the last KV entry
    // and leaves `hidden_a` = pre-output_norm hidden at position P-1).
    let t = std::time::Instant::now();
    let _ = gm.prefill_forward(&prompt[..prompt.len()-1], &mut state)
        .map_err(anyhow::Error::msg)?;
    let last_logits = gm.forward_token(*prompt.last().unwrap(), &mut state)
        .map_err(anyhow::Error::msg)?;
    println!("target prefill+1 = {:.1} ms ({} tokens)",
             t.elapsed().as_secs_f64() * 1e3, prompt.len());

    let mut next = argmax(&last_logits);
    let pos_const = state.pos - 1;
    println!("target argmax at position {pos_const}: {} ({:?})",
             next, tok.decode(&[next]));

    // Seed h_prev from the target's last hidden state.
    drafter.set_h_prev_from_target(&gm).map_err(anyhow::Error::msg)?;

    println!();
    println!("--- drafter proposals (k = {k}) ---");
    let mut prev_tok = next;
    let mut total = std::time::Duration::ZERO;
    for i in 0..k {
        let t = std::time::Instant::now();
        let logits = drafter.forward_step(&gm, &state, prev_tok, pos_const)
            .map_err(anyhow::Error::msg)?;
        let dt = t.elapsed();
        total += dt;
        next = argmax(&logits);
        println!("  step {i}: prev={prev_tok:>6} ({:?})  ->  drafted={next:>6} ({:?})  [{:.2} ms]",
                 tok.decode(&[prev_tok]),
                 tok.decode(&[next]),
                 dt.as_secs_f64() * 1e3);
        prev_tok = next;
    }
    println!("drafter mean: {:.2} ms/step over {k} steps",
             total.as_secs_f64() * 1e3 / k as f64);
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > best_v { best_v = x; best_i = i as u32; }
    }
    best_i
}

/// Standard temperature-softmax: subtract max for stability, exp, normalize.
/// `temperature` must be > 0 (caller checks).
fn softmax_with_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature;
    let mut max_v = f32::NEG_INFINITY;
    for &x in logits { if x > max_v { max_v = x; } }
    let mut out: Vec<f32> = logits.iter()
        .map(|&x| ((x - max_v) * inv_t).exp())
        .collect();
    let s: f32 = out.iter().sum();
    if s > 0.0 { for x in &mut out { *x /= s; } }
    out
}

/// Sample a token id from a logits vector by temperature-softmax.
fn sample_from_logits(logits: &[f32], temperature: f32,
                      rng: &mut reinstinct_engine::sampling::Rng) -> u32
{
    let p = softmax_with_temp(logits, temperature);
    sample_from_probs(&p, rng)
}

/// Sample from a vector of probabilities (must sum ~1.0).
fn sample_from_probs(probs: &[f32],
                     rng: &mut reinstinct_engine::sampling::Rng) -> u32
{
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc { return i as u32; }
    }
    (probs.len() - 1) as u32
}

/// Full speculative-decode generation loop (sequential verify). One
/// round: drafter proposes K tokens; target sequentially verifies via
/// greedy argmax acceptance; KV cache advances per accepted token; on
/// rejection target's own argmax replaces the drafted token. Always
/// commits ≥1 token per round.
///
/// With sequential verify the per-round cost is `K·drafter + (n_acc+1)·target`,
/// vs the K`drafter + 1`target a batched-verify path would hit. This is
/// a correctness path, not a speed path — see the MTP memory file.
fn mtp_gen_cli(target_path: &std::path::Path, drafter_path: &std::path::Path,
               prompt_text: Option<String>, system: Option<String>,
               k: usize, steps: usize, temperature: f32, seed: u64) -> anyhow::Result<()>
{
    use reinstinct_engine::chat::{ChatMessage, Role, format_gemma4};
    use reinstinct_engine::hip;
    use reinstinct_engine::model::gemma4::Gemma4Model;
    use reinstinct_engine::model::gemma4_assistant::Gemma4AssistantModel;
    use reinstinct_engine::runtime::{KernelCache, gemma4::{GpuGemma4, Gemma4GpuState}};
    use reinstinct_engine::runtime::gemma4_assistant::GpuGemma4Assistant;
    use reinstinct_engine::sampling::Rng;
    use reinstinct_engine::tokenizer::GemmaTokenizer;

    if k == 0 { anyhow::bail!("--k must be >= 1"); }
    let target_gguf  = GgufFile::open(target_path)?;
    let drafter_gguf = GgufFile::open(drafter_path)?;
    let tok = GemmaTokenizer::from_gguf(&target_gguf).map_err(anyhow::Error::msg)?;

    let prompt: Vec<u32> = if let Some(s) = &system {
        let user = prompt_text.clone().unwrap_or_default();
        let msgs = vec![
            ChatMessage { role: Role::System, content: s.clone() },
            ChatMessage { role: Role::User,   content: user },
        ];
        format_gemma4(&tok, &msgs, true).map_err(anyhow::Error::msg)?
    } else if let Some(t) = &prompt_text {
        let mut ids = vec![tok.bos_id];
        ids.extend(tok.encode(t));
        ids
    } else {
        anyhow::bail!("mtp-gen: pass --prompt or --system/--prompt");
    };

    if hip::device_count().ok().unwrap_or(0) < 1 { anyhow::bail!("no HIP device"); }
    let _dev = hip::Device::set(0).map_err(anyhow::Error::msg)?;
    let cache = KernelCache::new().map_err(anyhow::Error::msg)?;

    let target_model = Gemma4Model::load(&target_gguf).map_err(anyhow::Error::msg)?;
    let cfg_eos = target_model.config.eos_token_id;
    let max_seq = prompt.len() + steps + k + 16;
    let gm = GpuGemma4::new(&target_model, &target_gguf, &cache, max_seq)
        .map_err(anyhow::Error::msg)?;
    let drafter_model = Gemma4AssistantModel::load(&drafter_gguf).map_err(anyhow::Error::msg)?;
    let drafter = GpuGemma4Assistant::new(&drafter_model, &drafter_gguf, &gm, &cache)
        .map_err(anyhow::Error::msg)?;
    let mut state = Gemma4GpuState::new(&target_model, max_seq).map_err(anyhow::Error::msg)?;
    state.reset();

    println!("target = {} ({} tok prompt)", target_path.display(), prompt.len());
    println!("drafter = {}, K = {k}, steps = {steps}", drafter_path.display());

    // Initial prefill: process all prompt tokens; last forward leaves
    // `hidden_a` = hidden at last prompt position, and `verify_logits`
    // = target's prediction for the NEXT (un-validated) position.
    let t_pf = std::time::Instant::now();
    let _ = gm.prefill_forward(&prompt[..prompt.len()-1], &mut state)
        .map_err(anyhow::Error::msg)?;
    let mut verify_logits = gm.forward_token(*prompt.last().unwrap(), &mut state)
        .map_err(anyhow::Error::msg)?;
    println!("prefill = {:.0} ms", t_pf.elapsed().as_secs_f64() * 1e3);

    // The drafter is conditioned on (prev_tok, h_prev=target_hidden_at_pos).
    // After the initial forward_token, the natural last_token is the
    // FINAL prompt token (per HF: "input_ids[:, -1:]"), and h_prev is
    // target.last_hidden_state at that position. For subsequent rounds:
    // last_token = the last accepted/committed token.
    let mut last_tok = *prompt.last().unwrap();

    let mut generated: Vec<u32> = Vec::new();
    let mut total_drafted: usize = 0;
    let mut total_accepted: usize = 0;
    let mut hit_eos = false;
    let t_gen = std::time::Instant::now();
    let mut round_idx = 0usize;

    let sampling = temperature > 0.0;
    let mut rng = Rng::new(seed);

    while generated.len() < steps {
        round_idx += 1;
        // --- DRAFT phase ---
        // pos_const = state.pos - 1 is the position of the last validated
        // token. The drafter pins its position to that across the round.
        let pos_const = state.pos - 1;
        drafter.set_h_prev_from_target(&gm).map_err(anyhow::Error::msg)?;
        let mut drafted: Vec<u32> = Vec::with_capacity(k);
        // Drafter logits per step, kept around for the sampling-acceptance
        // ratio p_target/p_draft. Greedy mode doesn't read them.
        let mut drafter_logits_arr: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut prev = last_tok;
        for _ in 0..k {
            let logits_d = drafter.forward_step(&gm, &state, prev, pos_const)
                .map_err(anyhow::Error::msg)?;
            let d = if sampling {
                sample_from_logits(&logits_d, temperature, &mut rng)
            } else {
                argmax(&logits_d)
            };
            drafted.push(d);
            drafter_logits_arr.push(logits_d);
            prev = d;
        }

        // --- BATCHED VERIFY ---
        // Target processes the K drafted tokens in ONE forward at
        // positions [pre_verify_pos, pre_verify_pos+K), returning K
        // logit vectors. verify_logits_batch[i] predicts position
        // pre_verify_pos+i+1.
        let pre_verify_pos = state.pos;
        let verify_batch = gm.verify_forward(&drafted, &mut state)
            .map_err(anyhow::Error::msg)?;
        let _ = round_idx;

        // --- ACCEPTANCE ---
        // drafted[i] (at pos pre_verify_pos+i) is verified against:
        //   - argmax(verify_logits)            for i == 0   (target's prediction from BEFORE the round)
        //   - argmax(verify_batch[i-1])        for i >= 1   (target's prediction conditional on drafted[..i])
        // On first rejection: keep drafted[..i], commit target's pick at
        // pos pre_verify_pos+i, truncate the rejected slots.
        let mut accepted_this_round = 0usize;
        let mut rejected = false;
        for i in 0..drafted.len() {
            let predicting_logits = if i == 0 { &verify_logits } else { &verify_batch[i - 1] };
            let d = drafted[i];
            let (accept, replacement) = if sampling {
                // Rejection-sampling acceptance: accept drafted[i] with
                // probability min(1, p_target[d] / p_draft[d]); on reject
                // sample replacement from the residual (p_target -
                // p_draft)^+ , normalised.
                let p_t = softmax_with_temp(predicting_logits, temperature);
                let p_d = softmax_with_temp(&drafter_logits_arr[i], temperature);
                let r = rng.next_f32();
                let ratio = if p_d[d as usize] > 0.0 {
                    (p_t[d as usize] / p_d[d as usize]).min(1.0)
                } else { 0.0 };
                if r < ratio {
                    (true, d)
                } else {
                    let mut residual: Vec<f32> = p_t.iter().zip(p_d.iter())
                        .map(|(t, d)| (t - d).max(0.0))
                        .collect();
                    let s: f32 = residual.iter().sum();
                    if s > 0.0 { for x in &mut residual { *x /= s; } }
                    else { residual.copy_from_slice(&p_t); }
                    let repl = sample_from_probs(&residual, &mut rng);
                    (false, repl)
                }
            } else {
                let target_pred = argmax(predicting_logits);
                if d == target_pred { (true, d) } else { (false, target_pred) }
            };

            if accept {
                generated.push(d);
                accepted_this_round += 1;
                last_tok = d;
                if d == cfg_eos { hit_eos = true; rejected = true; break; }
                if generated.len() >= steps { rejected = true; break; }
            } else {
                state.truncate(pre_verify_pos + i);
                let new_verify = gm.forward_token(replacement, &mut state)
                    .map_err(anyhow::Error::msg)?;
                generated.push(replacement);
                last_tok = replacement;
                hit_eos = replacement == cfg_eos;
                verify_logits = new_verify;
                rejected = true;
                break;
            }
        }
        if !rejected {
            // All K accepted: verify_batch[K-1] is target's prediction
            // for the next position (= seed logits for the next round).
            verify_logits = verify_batch.last().cloned().unwrap();
        }
        total_drafted += drafted.len();
        total_accepted += accepted_this_round;
        let _ = round_idx;
        if hit_eos { break; }
    }

    let gen_secs = t_gen.elapsed().as_secs_f64();
    let n_gen = generated.len();
    println!();
    println!("--- generation ---");
    println!("{}", tok.decode(&generated));
    println!();
    println!("generated {n_gen} tokens in {:.2} s = {:.1} tok/s", gen_secs, n_gen as f64 / gen_secs);
    println!("draft accept rate: {} / {} = {:.0}%",
             total_accepted, total_drafted,
             100.0 * total_accepted as f64 / total_drafted.max(1) as f64);
    if hit_eos { println!("(hit EOS)"); }
    Ok(())
}

fn inspect(path: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    println!("file        = {}", path.display());
    println!("version     = {}", g.header.version);
    println!("tensors     = {}", g.header.tensor_count);
    println!("metadata    = {} kv pairs", g.header.metadata_kv_count);
    println!("alignment   = {}", g.alignment);
    println!("data_offset = {} bytes", g.data_section_offset);

    println!("\n--- metadata (scalars only) ---");
    for (k, v) in &g.metadata {
        match v {
            MetaValue::Array { element_type, values } => {
                println!("  {k} = <{:?}; {} entries>", element_type, values.len());
            }
            other => println!("  {k} = {}", short_value(other)),
        }
    }

    let mut hist: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let bytes = t.byte_size().unwrap_or(0);
        let e = hist.entry(format!("{:?}", t.ggml_type)).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    println!("\n--- tensor type histogram ---");
    let mut total_bytes = 0u64;
    for (k, (count, bytes)) in &hist {
        println!("  {k:8} {count:5} tensors  {:>10} MB", bytes / (1024 * 1024));
        total_bytes += bytes;
    }
    println!("  {:8} {:5}           {:>10} MB total",
        "", "", total_bytes / (1024 * 1024));

    if verbose {
        println!("\n--- all tensors ---");
        for t in &g.tensors {
            println!("  {:50} {:?} {:?}", t.name, t.ggml_type, t.shape());
        }
    } else {
        println!("\n--- top 10 tensors by size ---");
        let mut by_size: Vec<_> = g.tensors.iter().collect();
        by_size.sort_by_key(|t| std::cmp::Reverse(t.byte_size().unwrap_or(0)));
        for t in by_size.iter().take(10) {
            let mb = t.byte_size().unwrap_or(0) / (1024 * 1024);
            println!("  {:>4} MB  {:?}  {:?}  {}", mb, t.ggml_type, t.shape(), t.name);
        }
    }

    Ok(())
}

fn model(path: &std::path::Path) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    let arch = g.metadata_get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("<unknown>");
    println!("file = {}", path.display());
    println!("arch = {arch}");

    match arch {
        "qwen35" | "qwen35moe" => {
            let m = Qwen35Model::load(&g)?;
            print_qwen35(&m);
        }
        other => {
            anyhow::bail!("no typed loader for architecture {other:?} yet");
        }
    }
    Ok(())
}

fn print_qwen35(m: &Qwen35Model) {
    let c = &m.config;
    println!("\n--- config ---");
    println!("  blocks            = {}", c.block_count);
    println!("  hidden            = {}", c.hidden_size);
    println!("  ffn               = {}", c.ffn_size);
    println!("  vocab             = {}", c.vocab_size);
    println!("  context           = {}", c.context_length);
    println!("  rms_eps           = {:.2e}", c.rms_norm_eps);
    println!("  tied_embeddings   = {}", c.tied_embeddings);
    println!("  full attn:");
    println!("    n_heads         = {}", c.attn_n_heads);
    println!("    n_kv_heads      = {}", c.attn_n_kv_heads);
    println!("    head_dim        = {}", c.attn_head_dim);
    println!("  linear attn (GDN):");
    println!("    value_dim       = {}", c.gdn_value_dim);
    println!("    n_heads         = {}", c.gdn_n_heads);
    println!("    head_dim        = {}", c.gdn_head_dim);
    println!("    conv_kernel     = {}", c.gdn_conv_kernel);
    println!("  rope:");
    println!("    freq_base       = {}", c.rope_freq_base);
    println!("    rotated dims    = {} of {}", c.rope_dim_count, c.attn_head_dim);
    println!("    mrope sections  = {:?}", c.rope_dim_sections);
    println!("  layer schedule    = full attention every {} blocks", c.full_attention_interval);
    if let Some(moe) = &c.moe {
        println!("  MoE FFN:");
        println!("    experts         = {} ({} used/token)", moe.n_expert, moe.n_expert_used);
        println!("    expert_ff       = {}", moe.expert_ff);
        println!("    shared_expert_ff= {}", moe.shared_expert_ff);
    }

    println!("\n--- block schedule ---");
    for (i, &k) in m.block_kinds.iter().enumerate() {
        let tag = match k {
            BlockKind::LinearAttention => "L",
            BlockKind::FullAttention   => "F",
        };
        print!("  {i:2}:{tag}");
        if (i + 1) % 8 == 0 { println!(); }
    }
    if m.block_kinds.len() % 8 != 0 { println!(); }

    let n_full = m.block_kinds.iter().filter(|k| **k == BlockKind::FullAttention).count();
    let n_lin  = m.block_kinds.len() - n_full;
    println!("\n  total = {} linear, {} full", n_lin, n_full);
}

fn short_value(v: &MetaValue) -> String {
    match v {
        MetaValue::String(s) if s.len() > 80 => format!("{:?}…", &s[..80]),
        MetaValue::String(s) => format!("{s:?}"),
        MetaValue::U8(x)  => x.to_string(),  MetaValue::I8(x)  => x.to_string(),
        MetaValue::U16(x) => x.to_string(),  MetaValue::I16(x) => x.to_string(),
        MetaValue::U32(x) => x.to_string(),  MetaValue::I32(x) => x.to_string(),
        MetaValue::U64(x) => x.to_string(),  MetaValue::I64(x) => x.to_string(),
        MetaValue::F32(x) => format!("{x:.6}"),
        MetaValue::F64(x) => format!("{x:.6}"),
        MetaValue::Bool(b) => b.to_string(),
        MetaValue::Array { element_type, values } => {
            format!("<{:?}; {} entries>", element_type, values.len())
        }
    }
}
