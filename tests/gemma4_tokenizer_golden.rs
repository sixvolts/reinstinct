//! Gemma 4 tokenizer vs llama.cpp: encode a spread of texts with our
//! GemmaTokenizer and with tests/golden/tokenize (llama.cpp's vocab,
//! built by tests/golden/build.sh); the ids must match exactly.
//!
//!   cargo test --release --test gemma4_tokenizer_golden -- --ignored --nocapture

use std::io::Write;
use std::process::{Command, Stdio};

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::test_support;
use reinstinct_engine::tokenizer::GemmaTokenizer;

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
];

#[test]
#[ignore]
fn gemma_tokenizer_matches_llama_cpp() {
    let Some(path) = test_support::gemma_e4b_fixture() else { return };
    let tool = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/tokenize");
    if !std::path::Path::new(tool).exists() {
        eprintln!("skipping: build {tool} with tests/golden/build.sh");
        return;
    }
    let g = GgufFile::open(&path).expect("open");
    let tok = GemmaTokenizer::from_gguf(&g).unwrap();
    let mut bad = 0;
    for text in TEXTS {
        let mut child = Command::new(tool).arg(&path).stdin(Stdio::piped())
            .stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
        let out = child.wait_with_output().unwrap();
        let want: Vec<u32> = String::from_utf8(out.stdout).unwrap().trim().split(',')
            .filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
        let got = tok.encode(text);
        if got != want {
            bad += 1;
            eprintln!("MISMATCH {text:?}\n  ours  {got:?}\n  llama {want:?}");
        }
    }
    assert_eq!(bad, 0, "{bad} of {} texts tokenize differently from llama.cpp", TEXTS.len());
}
