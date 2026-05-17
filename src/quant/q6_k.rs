//! Q6_K: 6.5625 bpw symmetric K-quant.
//!
//! 210-byte block / 256 weights:
//!   u8    ql[128]      lower 4 bits of each 6-bit quant
//!   u8    qh[64]       upper 2 bits, four 2-bit pairs per byte
//!   i8    scales[16]   signed int8 sub-block scales
//!   fp16  d            super-block scale
//!
//! 16 sub-blocks of 16 weights each. Per-weight:
//!   q6 = (ql_nibble) | ((qh_pair) << 4)         ∈ 0..63 unsigned
//!   w  = d * scales[sub] * (q6 - 32)             symmetric, no `dmin`

use bytemuck::{Pod, Zeroable};

use crate::quant::half::{f16_to_f32, f32_to_f16};

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 210;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ6_K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: u16,
}

const _: () = assert!(std::mem::size_of::<BlockQ6_K>() == BYTES_PER_BLOCK);

pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);

    let blocks: &[BlockQ6_K] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);

    for (b, out_block) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d = f16_to_f32(b.d);

        // Each block splits into 2 outer chunks of 128 weights.
        // Per chunk: 4 sub-rows of 32 outputs each, indexed as:
        //   y[l + 0/32/64/96] for l in 0..32
        // qh contributes 4 separate 2-bit fields per byte:
        //   bits 0..1 → +0, bits 2..3 → +32, bits 4..5 → +64, bits 6..7 → +96
        // Scale index `is = l / 16`, advanced across chunks.
        for chunk in 0..2 {
            let ql_off = chunk * 64;
            let qh_off = chunk * 32;
            let sc_off = chunk * 8;
            let y_off  = chunk * 128;

            for l in 0..32usize {
                let is = l / 16; // 0 for l<16, 1 for l>=16
                let qh = b.qh[qh_off + l];
                let q1 = ((b.ql[ql_off + l]      & 0x0F) | (((qh >> 0) & 0x3) << 4)) as i32 - 32;
                let q2 = ((b.ql[ql_off + l + 32] & 0x0F) | (((qh >> 2) & 0x3) << 4)) as i32 - 32;
                let q3 = ((b.ql[ql_off + l]      >> 4)   | (((qh >> 4) & 0x3) << 4)) as i32 - 32;
                let q4 = ((b.ql[ql_off + l + 32] >> 4)   | (((qh >> 6) & 0x3) << 4)) as i32 - 32;

                out_block[y_off + l]      = d * b.scales[sc_off + is]     as f32 * q1 as f32;
                out_block[y_off + l + 32] = d * b.scales[sc_off + is + 2] as f32 * q2 as f32;
                out_block[y_off + l + 64] = d * b.scales[sc_off + is + 4] as f32 * q3 as f32;
                out_block[y_off + l + 96] = d * b.scales[sc_off + is + 6] as f32 * q4 as f32;
            }
        }
    }
}

/// Repack a Q6_K matvec weight `[out_dim, in_dim]` into three contiguous
/// planes (row stride `nsp = q4_k::repacked_n_sub_padded(in_dim)`,
/// 32-weight sub-blocks). Q6_K is symmetric (`q-32`, no `dmin`) with one
/// signed int8 scale per 16 weights — so each 32-weight sub-block holds
/// two pre-multiplied scales.
///   * nibble plane — `nsp*16` B/row, low 4 bits per weight, permuted as
///     in `q4_k::repack_for_matvec`.
///   * high-2-bit plane — `nsp*8` B/row. Per sub-block 8 bytes: byte `g`
///     holds the four 2-bit fields of dp4a group `g` (weight `b` at bits
///     `2b..2b+1`).
///   * scale plane — `nsp*4` B/row: `fp16(d·sc_lo)`, `fp16(d·sc_hi)`
///     (sc_lo for weights 0..15, sc_hi for 16..31).
pub fn repack_for_matvec(bytes: &[u8], in_dim: usize, out_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % BLOCK_SIZE, 0, "Q6_K in_dim must be a multiple of 256");
    let n_blocks = in_dim / BLOCK_SIZE;
    let nsp      = crate::quant::q4_k::repacked_n_sub_padded(in_dim);
    let nib_len  = out_dim * nsp * 16;
    let h2_len   = out_dim * nsp * 8;
    let mut out  = vec![0u8; nib_len + h2_len + out_dim * nsp * 4];

    let blocks: &[BlockQ6_K] =
        bytemuck::cast_slice(&bytes[..out_dim * n_blocks * BYTES_PER_BLOCK]);

    for row in 0..out_dim {
        for blk in 0..n_blocks {
            let b = &blocks[row * n_blocks + blk];
            let d = f16_to_f32(b.d);

            for s in 0..8usize {                 // 8 sub-blocks of 32 per 256-block
                let gsb   = blk * 8 + s;
                let chunk = s / 4;
                let quad  = s % 4;
                let ql_off = chunk * 64;
                let qh_off = chunk * 32;

                let mut lo = [0u8; 32];
                let mut hi = [0u8; 32];
                for l in 0..32 {
                    let qh = b.qh[qh_off + l];
                    let (low4, h2) = match quad {
                        0 => (b.ql[ql_off + l]      & 0x0F,  qh        & 3),
                        1 => (b.ql[ql_off + l + 32] & 0x0F, (qh >> 2)  & 3),
                        2 => (b.ql[ql_off + l]      >> 4,   (qh >> 4)  & 3),
                        _ => (b.ql[ql_off + l + 32] >> 4,   (qh >> 6)  & 3),
                    };
                    lo[l] = low4;
                    hi[l] = h2;
                }

                let nib_off = (row * nsp + gsb) * 16;
                let mut h2p = [0u8; 8];
                for j in 0..4 {
                    for bb in 0..4 {
                        out[nib_off + j * 4 + bb] =
                            lo[4 * j + bb] | (lo[16 + 4 * j + bb] << 4);
                        h2p[2 * j]     |= hi[4 * j + bb]      << (2 * bb);
                        h2p[2 * j + 1] |= hi[16 + 4 * j + bb] << (2 * bb);
                    }
                }
                let h2_off = nib_len + (row * nsp + gsb) * 8;
                out[h2_off..h2_off + 8].copy_from_slice(&h2p);

                let sc_lo = b.scales[chunk * 8 + quad * 2]     as f32;
                let sc_hi = b.scales[chunk * 8 + quad * 2 + 1] as f32;
                let so = nib_len + h2_len + (row * nsp + gsb) * 4;
                out[so..so + 2].copy_from_slice(&f32_to_f16(d * sc_lo).to_le_bytes());
                out[so + 2..so + 4].copy_from_slice(&f32_to_f16(d * sc_hi).to_le_bytes());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    /// Build a Q6_K block where every weight has the same 6-bit value `q`
    /// (so all outputs = d * scale_for_that_subblock * (q - 32)).
    fn synth_block_uniform_q(d: f32, scale: i8, q: u8) -> Vec<u8> {
        assert!(q < 64);
        let mut bytes = vec![0u8; BYTES_PER_BLOCK];

        let ql_nibble = q & 0x0F;
        let ql_byte = ql_nibble | (ql_nibble << 4);
        for i in 0..128 {
            bytes[i] = ql_byte;
        }

        let pair = (q >> 4) & 0x3;
        let qh_byte = pair | (pair << 2) | (pair << 4) | (pair << 6);
        for i in 0..64 {
            bytes[128 + i] = qh_byte;
        }

        for i in 0..16 {
            bytes[192 + i] = scale as u8;
        }
        bytes[208..210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        bytes
    }

    #[test]
    fn zero_centered_at_q_equals_32() {
        // q=32 → w = d * sc * 0 = 0 regardless of d, sc.
        let bytes = synth_block_uniform_q(2.5, -7, 32);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn unit_scale_returns_signed_q_minus_32() {
        // d=1, scale=1, q=k → output all = (k - 32)
        for q in [0u8, 16, 31, 32, 33, 47, 63] {
            let bytes = synth_block_uniform_q(1.0, 1, q);
            let mut out = vec![0.0_f32; 256];
            dequantize_to_f32(&bytes, &mut out);
            let expected = q as f32 - 32.0;
            assert!(out.iter().all(|v| *v == expected),
                "q={q}: first values {:?}", &out[..4]);
        }
    }

    #[test]
    fn negative_scale_flips_sign() {
        // d=1, scale=-3, q=33 → output = 1 * -3 * 1 = -3
        let bytes = synth_block_uniform_q(1.0, -3, 33);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == -3.0));
    }

    #[test]
    fn per_subblock_scale_distinguishes_outputs() {
        // d=1, sub-block scales[0..16] = [0,1,2,3,...,15], q=33 (so q-32=1).
        // Output[i] for i in sub-block sb should equal scales[sb].
        let mut bytes = synth_block_uniform_q(1.0, 0, 33);
        for i in 0..16 {
            bytes[192 + i] = i as i8 as u8;
        }
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        for sb in 0..16 {
            for w in 0..16 {
                let idx = sb * 16 + w;
                // The output position for sub-block `sb` is determined by the
                // chunk + (is, +0/+32/+64/+96) + l layout. Walk through the
                // dequant order to predict which scale was used.
                let chunk = idx / 128;
                let inner = idx % 128;
                let group = inner / 32;       // 0..=3 → +0,+32,+64,+96
                let l = inner % 32;
                let is = l / 16;
                let sc_idx = chunk * 8 + is + group * 2;
                assert_eq!(out[idx], sc_idx as f32,
                    "idx={idx} chunk={chunk} group={group} l={l} is={is} expected scales[{sc_idx}]");
            }
        }
    }
}
