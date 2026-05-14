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
    }
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
    let m = Qwen35F32Model::load(&g)?;
    let cfg = &m.model.config;
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
    let mut sum_block = vec![0.0_f32; m.model.block_kinds.len()];
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
    for (acc, &kind) in sum_block.iter().zip(m.model.block_kinds.iter()) {
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

    // CPU baseline for direct comparison.
    println!("\n--- CPU forward_token, {iters} iterations ---");
    let mut cpu_state = m.new_state(iters + 4);
    let _ = m.forward_token(token, &mut cpu_state);  // warmup
    cpu_state.reset();
    let mut cpu_times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let _ = m.forward_token(token, &mut cpu_state);
        cpu_times_us.push(t.elapsed().as_micros() as u64);
        cpu_state.reset();
    }
    cpu_times_us.sort_unstable();
    let cpu_median = cpu_times_us[cpu_times_us.len() / 2] as f64 / 1000.0;
    println!("  median  {cpu_median:>8.3} ms  ({:>5.1} tok/s)", 1000.0 / cpu_median);
    let speedup = cpu_median / median;
    let label = if speedup >= 1.0 { "speedup" } else { "slowdown" };
    println!("\nGPU vs CPU: {speedup:.2}× {label} (median)");
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
        let gpu = GpuQwen35::new(&m, &g, &cache, max_seq).map_err(anyhow::Error::msg)?;
        println!("weights load  = {:.2} s", t_load.elapsed().as_secs_f32());
        let mut state = Qwen35GpuState::new(&m, max_seq).map_err(anyhow::Error::msg)?;
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
