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
        Command::GenerateText { path, prompt, tokens, steps, temperature, top_k, seed, gpu } =>
            generate_text(&path, prompt, tokens, steps, temperature, top_k, seed, gpu),
    }
}

fn generate_text(path: &std::path::Path, prompt_text: Option<String>,
                 tokens: Option<Vec<u32>>, steps: usize,
                 temperature: f32, top_k: usize, seed: u64, gpu: bool) -> anyhow::Result<()> {
    use reinstinct_engine::sampling::{Rng, sample_temp_topk};
    use reinstinct_engine::tokenizer::Tokenizer;

    let g = GgufFile::open(path)?;
    let arch = g.metadata_get("general.architecture")
        .and_then(|v| v.as_str()).unwrap_or("<unknown>");
    if arch == "gemma4" {
        return generate_text_gemma4(&g, path, prompt_text, tokens, steps,
                                    temperature, top_k, seed, gpu);
    }
    // Typed model — config + quantized tensor refs only. The f32
    // CPU oracle (Qwen35F32Model) is loaded lazily in the CPU branch;
    // building it for --gpu would needlessly materialise the whole
    // model in host RAM (87 GB+ on a 27B model — OOM).
    let model = Qwen35Model::load(&g)?;
    let cfg = &model.config;

    // Prompt resolution: --prompt text (BPE-encoded) > --tokens > [EOS].
    let prompt: Vec<u32> = if let Some(text) = &prompt_text {
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
        // Decode timer — steady-state per-token cost, excluding the
        // one-time weight load and the prompt prefill.
        let t_dec = std::time::Instant::now();
        for _ in 0..steps {
            let tok = sample_temp_topk(&logits, temperature, top_k, &mut rng);
            all.push(tok);
            if tok == cfg.eos_token_id { break; }
            logits = gpu.forward_token(tok, &mut state).map_err(anyhow::Error::msg)?;
        }
        let n_dec = all.len() - prompt.len();
        if n_dec > 0 {
            let d = t_dec.elapsed().as_secs_f64();
            println!("decode        = {:.1} ms/token ({:.1} tok/s) over {n_dec} forwards",
                d * 1e3 / n_dec as f64, n_dec as f64 / d);
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
                        tokens: Option<Vec<u32>>, steps: usize,
                        temperature: f32, top_k: usize, seed: u64, gpu: bool) -> anyhow::Result<()> {
    use reinstinct_engine::sampling::{Rng, sample_temp_topk};
    use reinstinct_engine::cpu::gemma4::Gemma4CpuModel;
    use reinstinct_engine::model::gemma4::Gemma4Model;
    use reinstinct_engine::tokenizer::GemmaTokenizer;

    let cfg_eos = Gemma4Model::load(g).map_err(anyhow::Error::msg)?.config.eos_token_id;
    // Gemma 4 SentencePiece tokenizer — encodes --prompt text and
    // decodes the generated ids back to text.
    let tok = GemmaTokenizer::from_gguf(g).ok();
    let prompt: Vec<u32> = if let Some(text) = &prompt_text {
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
            let lg = gm.prefill_forward(&cache, &prompt, &mut state)
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
        let use_graph = std::env::var_os("REINSTINCT_NO_GRAPH").is_none();
        let t_cap = std::time::Instant::now();
        let graph = gm.capture_forward_graph(&state).map_err(anyhow::Error::msg)?;
        if use_graph {
            println!("graph capture = {:.2} s", t_cap.elapsed().as_secs_f32());
        } else {
            println!("backend mode  = per-kernel (REINSTINCT_NO_GRAPH)");
        }
        let fwd = |gm: &GpuGemma4, t: u32, st: &mut Gemma4GpuState| {
            if use_graph { gm.forward_via_graph(&graph, t, st) }
            else { gm.forward_token(t, st) }
        };
        // Prefill the prompt in one batched pass — this populates every
        // layer's KV cache, so decode continues straight from position P.
        let t_prefill = std::time::Instant::now();
        let mut lg = gm.prefill_forward(&cache, &prompt, &mut state)
            .map_err(anyhow::Error::msg)?;
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
    let exec = gpu.capture_forward_graph(token, &mut state).map_err(anyhow::Error::msg)?;
    println!("\nHIP graph capture + instantiate took {:.3} ms",
        t_cap.elapsed().as_secs_f64() * 1000.0);
    state.reset().map_err(anyhow::Error::msg)?;
    let _ = gpu.forward_token_via_graph(&exec, &mut state).map_err(anyhow::Error::msg)?;  // warmup
    state.reset().map_err(anyhow::Error::msg)?;

    let mut g_times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let _ = gpu.forward_token_via_graph(&exec, &mut state).map_err(anyhow::Error::msg)?;
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
        "qwen35" => {
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
