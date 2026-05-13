use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use reinstinct_engine::gguf::{GgufFile, MetaValue};

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
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Command::Inspect { path, verbose } => inspect(&path, verbose),
    }
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
