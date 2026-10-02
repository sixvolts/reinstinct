//! Diagnostic: mean next-token NLL of the Qwen decode loop over token
//! sequences in REINSTINCT_PPL_SEQS (one comma-separated sequence per
//! line, e.g. from tests/golden/tokenize), for comparison against
//! tests/golden/nll on the same ids.
//!   REINSTINCT_GGUF_FIXTURE=... REINSTINCT_PPL_SEQS=... cargo test \
//!       --release --test qwen35_ppl -- --ignored --nocapture

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::hip;
use reinstinct_engine::model::qwen3_5::Qwen35Model;
use reinstinct_engine::runtime::KernelCache;
use reinstinct_engine::runtime::pipeline::Qwen35Pipeline;
use reinstinct_engine::test_support;

#[test]
#[ignore]
fn qwen_decode_perplexity() {
    let Ok(file) = std::env::var("REINSTINCT_PPL_SEQS") else { return };
    let (Some(path), Some(())) = (test_support::qwen_fixture(), test_support::gpu()) else { return };
    let seqs: Vec<Vec<u32>> = std::fs::read_to_string(file).unwrap().lines()
        .map(|l| l.split(',').filter(|t| !t.is_empty()).map(|t| t.trim().parse().unwrap()).collect())
        .filter(|s: &Vec<u32>| s.len() > 1).collect();
    let max_seq = seqs.iter().map(Vec::len).max().unwrap() + 8;
    hip::Device::set(0).unwrap();
    let g = GgufFile::open(&path).expect("open");
    let model = Qwen35Model::load(&g).expect("load");
    let cache = KernelCache::new().unwrap();
    let gm = Qwen35Pipeline::new(&model, &g, &cache, max_seq, &[0], None).unwrap();
    let mut state = gm.new_state(&model, max_seq).unwrap();
    let (mut nll, mut n) = (0f64, 0usize);
    for s in &seqs {
        state.reset().unwrap();
        for w in s.windows(2) {
            let l = gm.forward_token(w[0], &mut state).unwrap();
            let mx = l.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let lse = mx + l.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
            nll += lse - l[w[1] as usize] as f64;
            n += 1;
        }
    }
    eprintln!("{}: {n} tokens, mean NLL {:.4}, ppl {:.3}", path.display(), nll / n as f64, (nll / n as f64).exp());
}
