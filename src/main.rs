use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "reinstinct-engine", version, about)]
struct Cli {
    /// Path to a GGUF model file
    #[arg(short, long)]
    model: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(path) = cli.model {
        println!("model: {}", path.display());
    } else {
        println!("reinstinct-engine v{} — pass --model <path.gguf>", env!("CARGO_PKG_VERSION"));
    }
    Ok(())
}
