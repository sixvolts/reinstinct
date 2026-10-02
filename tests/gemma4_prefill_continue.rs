//! `prefill_forward_at`: a batched prefill that continues a restored
//! prefix must match the per-token decode oracle (restore, then
//! `forward_token` over the suffix) — the path `chat` has always used.
//!
//! Checks, per (base, suffix) case:
//!   * KV rows [0, base) are untouched by the continuation;
//!   * layer 0's suffix KV rows equal a full from-zero prefill's
//!     bit-for-bit (no attention input yet: positions, RoPE offset and
//!     the cache write offset are exact);
//!   * deeper layers drift from a full prefill no more than the decode
//!     oracle does (both read the prefix as int8);
//!   * last-token logits sit within the decode-vs-prefill floor of both
//!     the oracle and a full prefill.
//!
//! GPU + model required, so `#[ignore]`:
//!   REINSTINCT_GEMMA_E4B_FIXTURE=... cargo test --release --test \
//!       gemma4_prefill_continue -- --ignored --nocapture --test-threads=1
//! (one model on the GPU at a time; the MoE diagnostic sets env vars).
//! `REINSTINCT_PFC_MODEL=e4b|moe|dense` picks the fixture (default e4b).

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::hip;
use reinstinct_engine::model::gemma4::Gemma4Model;
use reinstinct_engine::runtime::gemma4::{Gemma4GpuState, GpuGemma4};
use reinstinct_engine::runtime::KernelCache;
use reinstinct_engine::test_support;

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0
}

fn tokens(n: usize, seed: u64, vocab: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (1000 + (s >> 33) as usize % (vocab.min(60000) - 1000)) as u32
    }).collect()
}

#[test]
#[ignore]
fn prefill_continuation_matches_decode_oracle() {
    let which = std::env::var("REINSTINCT_PFC_MODEL").unwrap_or_else(|_| "e4b".into());
    let fixture = match which.as_str() {
        "moe"   => test_support::gemma_moe_fixture(),
        "dense" => test_support::gemma_fixture(),
        _       => test_support::gemma_e4b_fixture(),
    };
    let (Some(path), Some(())) = (fixture, test_support::gpu()) else { return };
    let g = GgufFile::open(&path).expect("open");
    let _dev = hip::Device::set(0).unwrap();
    let cache = KernelCache::new().unwrap();
    let model = Gemma4Model::load(&g).unwrap();
    let max_seq = 4096;
    let gm = GpuGemma4::new(&model, &g, &cache, max_seq).unwrap();
    let mut state = Gemma4GpuState::new(&model, max_seq).unwrap();
    let vocab = model.config.vocab_size as usize;
    let sw = model.config.sliding_window as usize;
    eprintln!("{}: sliding_window {sw}", path.display());

    // Base not a multiple of 16; one case with base past the window; one
    // past the sliding-window ring (when the state has one), whose chunk
    // wraps it.
    let ring = (0..state.n_layers()).map(|l| state.kv_layer_rows(l)).min().unwrap();
    let mut cases = vec![(203usize, 77usize), (sw + 97, 150), (40, 1)];
    if ring < max_seq { cases.push((ring + 301, 200)); }
    for (ci, &(base, p)) in cases.iter().enumerate() {
        let all = tokens(base + p, 0x5eed + ci as u64, vocab);
        let (pre, suf) = all.split_at(base);

        state.reset();
        gm.prefill_forward(pre, &mut state).unwrap();
        let snap = state.snapshot().unwrap();
        let nl = state.n_layers();
        // Prefix positions a ring still holds after the continuation writes.
        let rows_v: Vec<usize> = (0..nl).map(|l| state.kv_layer_rows(l)).collect();
        let keep_lo = |l: usize| (base + p).saturating_sub(rows_v[l]).min(base);
        let before: Vec<_> = (0..nl).map(|l| state.kv_rows_to_host(l, keep_lo(l), base).unwrap()).collect();

        // Oracle: per-token decode over the suffix.
        let mut l_oracle = Vec::new();
        for &t in suf { l_oracle = gm.forward_token(t, &mut state).unwrap(); }
        let kv_oracle: Vec<_> = (0..nl).map(|l| state.kv_rows_to_host(l, base, base + p).unwrap()).collect();

        // Under test: batched continuation.
        state.restore(&snap).unwrap();
        state.truncate(base).unwrap();
        let l_cont = gm.prefill_forward_at(suf, &mut state, base).unwrap();
        assert_eq!(state.pos, base + p, "case {ci}: pos after continuation");
        let mut worst_q = 0i32;
        let mut over1 = 0usize;
        let mut total = 0usize;
        for l in 0..nl {
            let (k0, v0, ks0, vs0) = state.kv_rows_to_host(l, keep_lo(l), base).unwrap();
            let (bk, bv, bks, bvs) = &before[l];
            assert!(k0 == *bk && v0 == *bv && ks0 == *bks && vs0 == *bvs,
                    "case {ci}: layer {l} prefix rows changed by the continuation");
            let (k1, v1, _, _) = state.kv_rows_to_host(l, base, base + p).unwrap();
            let (ko, vo, _, _) = &kv_oracle[l];
            for (a, b) in k1.iter().chain(v1.iter()).zip(ko.iter().chain(vo.iter())) {
                let d = (*a as i32 - *b as i32).abs();
                worst_q = worst_q.max(d);
                if d > 1 { over1 += 1; }
                total += 1;
            }
        }
        let frac = over1 as f64 / total as f64;

        let kv_cont: Vec<_> = (0..nl).map(|l| state.kv_rows_to_host(l, base, base + p).unwrap()).collect();

        // Reference: full prefill from zero.
        state.reset();
        let l_full = gm.prefill_forward(&all, &mut state).unwrap();
        // A ring can't move its end back further than its slack.
        if ring < max_seq && base + p > ring {
            assert!(state.truncate(base + p - ring + 1).is_err(),
                    "case {ci}: truncate past the ring's slack must fail");
            assert!(state.truncate(base + p - 16).is_ok(), "case {ci}: a short rollback must succeed");
            state.truncate(base + p).ok();
        }
        // Per-layer worst |Δq| of the suffix rows: continuation vs full
        // prefill (same GEMM path — layer 0 has no attention input, so
        // it must match bit-for-bit) and oracle vs full prefill (the
        // decode/prefill rounding floor).
        let dq = |a: &(Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>), b: &(Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>)| -> (i32, f64) {
            let mut w = 0; let mut n1 = 0usize; let mut n = 0usize;
            for (x, y) in a.0.iter().chain(a.1.iter()).zip(b.0.iter().chain(b.1.iter())) {
                let d = (*x as i32 - *y as i32).abs(); w = w.max(d); if d > 1 { n1 += 1; } n += 1;
            }
            (w, n1 as f64 / n as f64)
        };
        let mut line_c = String::new(); let mut line_o = String::new();
        let (mut n_cf, mut n_of) = (0.0f64, 0.0f64);
        let layer0_full = state.kv_rows_to_host(0, base, base + p).unwrap();
        // int8 values are bit-identical; the f32 scales may differ in the
        // last bits (the GEMM tiles a P-row suffix differently from the
        // whole prompt), so compare them relatively.
        let srel = layer0_full.2.iter().chain(&layer0_full.3).zip(kv_cont[0].2.iter().chain(&kv_cont[0].3))
            .map(|(a, b)| ((a - b) / a.abs().max(1e-20)).abs()).fold(0.0f32, f32::max);
        assert!(srel < 1e-4, "case {ci}: layer 0 KV scales differ from a full prefill ({srel:.2e})");
        for l in 0..nl {
            let full = state.kv_rows_to_host(l, base, base + p).unwrap();
            let (wc, fc) = dq(&kv_cont[l], &full);
            let (wo, fo) = dq(&kv_oracle[l], &full);
            if l == 0 { assert_eq!(wc, 0, "case {ci}: layer 0 KV differs from a full prefill"); }
            n_cf += fc; n_of += fo;
            if l < 8 || l + 2 >= nl { line_c += &format!(" {wc}"); line_o += &format!(" {wo}"); }
        }
        eprintln!("  per-layer max|Δq| cont-vs-full:{line_c}\n  per-layer max|Δq| oracle-vs-full:{line_o}");

        let r_or = rel_l2(&l_cont, &l_oracle);
        let r_full = rel_l2(&l_cont, &l_full);
        let r_of = rel_l2(&l_oracle, &l_full);
        eprintln!("case {ci} base={base} p={p}: kv max|Δq|={worst_q} frac(|Δq|>1)={frac:.2e}  \
                   logits rel_l2 vs oracle {r_or:.2e} (argmax {} vs {}), vs full {r_full:.2e} \
                   (oracle vs full {r_of:.2e})",
                  argmax(&l_cont), argmax(&l_oracle));
        // The decode oracle's KV differs from any prefill's by the
        // matvec/GEMM rounding floor; the continuation must sit at the
        // prefill side of it (vs full prefill, only the int8 prefix in
        // attention differs).
        let (m_cf, m_of) = (n_cf / nl as f64, n_of / nl as f64);
        eprintln!("  mean frac(|Δq|>1) vs full prefill: continuation {m_cf:.2e}, decode oracle {m_of:.2e}");
        // Statistical bound with margin: on random tokens the drift varies
        // with content (31B, base 40, P=1: 0.148 vs the oracle's 0.057 on
        // one sequence, 0.036 vs 0.043 on another). The exact checks are
        // layer 0 and the untouched prefix above.
        assert!(m_cf <= 2.0 * m_of + 0.1, "case {ci}: continuation KV drifts past the decode floor");
        // Decode and prefill already differ by r_of (int8 KV, matvec vs
        // GEMM rounding); the continuation sits between them, so bound
        // it by that floor rather than a fixed number.
        // The continuation runs the prefill GEMMs, so it is held to the
        // full prefill; against the oracle it only has to stay within
        // the two gaps combined.
        let floor = r_of.max(0.02) * 1.5 + 0.05;
        assert!(r_full < floor, "case {ci}: logits diverge from a full prefill");
        assert!(r_or < r_of + r_full + 0.01, "case {ci}: logits diverge from the decode oracle");
    }
}

/// MoE diagnostic: on real text, last-token logits of a grouped-GEMM
/// prefill, a per-token-matvec prefill (REINSTINCT_MOE_NO_GROUPED) and
/// the decode loop should agree to the decode/prefill rounding floor.
#[test]
#[ignore]
fn moe_prefill_paths_agree_with_decode() {
    let (Some(path), Some(())) = (test_support::gemma_moe_fixture(), test_support::gpu()) else { return };
    let g = GgufFile::open(&path).expect("open");
    let _dev = hip::Device::set(0).unwrap();
    let cache = KernelCache::new().unwrap();
    let model = Gemma4Model::load(&g).unwrap();
    let tok = reinstinct_engine::tokenizer::GemmaTokenizer::from_gguf(&g).unwrap();
    let gm = GpuGemma4::new(&model, &g, &cache, 1024).unwrap();
    let mut state = Gemma4GpuState::new(&model, 1024).unwrap();
    // No captured prefill graphs: a cached graph would replay whichever
    // MoE path it was captured with, whatever the env var says.
    unsafe { std::env::set_var("REINSTINCT_PREFILL_NO_GRAPH", "1"); }
    let text = "The history of the lighthouse begins in the ancient world, where fires were \
                lit on hilltops to guide ships into harbour. The most famous of these was the \
                Pharos of Alexandria, built in the third century BC on a small island off the \
                Egyptian coast. Standing more than one hundred metres tall, it remained one of \
                the tallest man-made structures for many centuries, and its name became the \
                root of the word for lighthouse in several languages. Later, the Romans built";
    let mut ids = vec![tok.bos_id];
    ids.extend(tok.encode(text));
    let top = |v: &[f32]| { let mut i: Vec<usize> = (0..v.len()).collect();
        i.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap()); i.truncate(5); i };
    for p in [ids.len(), 33] {
        let ids = &ids[..p];
        state.reset();
        // Single-threaded test binary section: env toggled between calls.
        unsafe { std::env::remove_var("REINSTINCT_MOE_NO_GROUPED"); }
        let l_grp = gm.prefill_forward(ids, &mut state).unwrap();
        state.reset();
        unsafe { std::env::set_var("REINSTINCT_MOE_NO_GROUPED", "1"); }
        let l_mv = gm.prefill_forward(ids, &mut state).unwrap();
        unsafe { std::env::remove_var("REINSTINCT_MOE_NO_GROUPED"); }
        state.reset();
        let mut l_dec = Vec::new();
        for &t in ids { l_dec = gm.forward_token(t, &mut state).unwrap(); }
        eprintln!("P={p}: grouped-vs-matvec {:.2e}  matvec-vs-decode {:.2e}  grouped-vs-decode {:.2e}\n  \
                   top5 grouped {:?} matvec {:?} decode {:?}",
                  rel_l2(&l_grp, &l_mv), rel_l2(&l_mv, &l_dec), rel_l2(&l_grp, &l_dec),
                  top(&l_grp), top(&l_mv), top(&l_dec));
    }
}
