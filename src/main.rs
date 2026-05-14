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
    /// Run the CPU forward pass on a single input token, print top-K logits.
    Generate {
        path: PathBuf,
        /// Input token id. Defaults to the model's EOS token id from metadata.
        #[arg(short, long)]
        token: Option<u32>,
        /// Number of top logits to print.
        #[arg(short, long, default_value_t = 10)]
        k: usize,
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
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Command::Inspect { path, verbose } => inspect(&path, verbose),
        Command::Model { path } => model(&path),
        Command::Generate { path, token, k } => generate(&path, token, k),
        Command::DebugEmbed { path, tokens } => debug_embed(&path, &tokens),
        Command::Bench { path, iters, token } => bench(&path, iters, token),
    }
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

fn generate(path: &std::path::Path, token: Option<u32>, k: usize) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    let m = Qwen35F32Model::load(&g)?;
    let cfg = &m.model.config;
    let token = token.unwrap_or(cfg.eos_token_id);
    println!("model         = {}", path.display());
    println!("vocab         = {}", cfg.vocab_size);
    println!("input token   = {token}");

    let mut state = m.new_state(16);
    let t0 = std::time::Instant::now();
    let logits = m.forward_token(token, &mut state);
    println!("forward took  = {:.2} s", t0.elapsed().as_secs_f32());

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
