//! Tokenizers vs llama.cpp: encode a spread of texts with our
//! GemmaTokenizer / Qwen Tokenizer and with tests/golden/tokenize
//! (llama.cpp's vocab, built by tests/golden/build.sh); the ids must
//! match exactly.
//!
//!   cargo test --release --test tokenizer_golden -- --ignored --nocapture

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::test_support;
use reinstinct_engine::tokenizer::{GemmaTokenizer, Tokenizer};

const TEXTS: &[&str] = &[
    "Write a detailed history of the printing press, from Gutenberg to the digital age.",
    "Hello world.\n\nThis is a  test of\nnewlines.",
    "  leading spaces and trailing   ",
    "\n\n\nstarts with newlines\n",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "Ünïcödé — “quotes”, em-dashes, emoji 🙂🚀, CJK 漢字かな, Arabic مرحبا.",
    "tabs\tand\r\nwindows line endings\r\n",
    "numbers 1234567890 3.14159 1e-9 0xDEADBEEF",
    "<|turn>user literal markup is plain text here<turn|>",
    "Combining marks: नमस्ते दुनिया, สวัสดีครับ, café vs cafe\u{301}, Ελληνικά.",
    "it's they'LL we've I'm don't 'quoted' -- ... !!!",
];

/// llama.cpp's ids for `text` with the vocab in `gguf`.
fn reference(tool: &str, gguf: &Path, text: &str) -> Vec<u32> {
    let mut child = Command::new(tool).arg(gguf).stdin(Stdio::piped())
        .stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    String::from_utf8(out.stdout).unwrap().trim().split(',')
        .filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect()
}

fn compare(gguf: &Path, encode: impl Fn(&str) -> Vec<u32>) {
    let tool = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/tokenize");
    if !Path::new(tool).exists() {
        eprintln!("skipping: build {tool} with tests/golden/build.sh");
        return;
    }
    let mut bad = 0;
    for text in TEXTS {
        let want = reference(tool, gguf, text);
        let got = encode(text);
        if got != want {
            bad += 1;
            eprintln!("MISMATCH {text:?}\n  ours  {got:?}\n  llama {want:?}");
        }
    }
    assert_eq!(bad, 0, "{bad} of {} texts tokenize differently from llama.cpp", TEXTS.len());
}

#[test]
#[ignore]
fn gemma_tokenizer_matches_llama_cpp() {
    let Some(path) = test_support::gemma_e4b_fixture() else { return };
    let g = GgufFile::open(&path).expect("open");
    let tok = GemmaTokenizer::from_gguf(&g).unwrap();
    compare(&path, |t| tok.encode(t));
}

#[test]
#[ignore]
fn qwen_tokenizer_matches_llama_cpp() {
    let Some(path) = test_support::qwen_fixture() else { return };
    let g = GgufFile::open(&path).expect("open");
    let tok = Tokenizer::from_gguf(&g).unwrap();
    compare(&path, |t| tok.encode(t));
}
