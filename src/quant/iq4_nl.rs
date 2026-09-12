//! IQ4_NL: 4.5 bpw non-linear 4-bit, block of 32.
//!
//! 18-byte block / 32 weights:
//!   fp16  d          block scale
//!   u8    qs[16]     32 nibbles indexing the 16-entry `KVALUES_IQ4NL` codebook
//!
//! `w = d * KVALUES[nibble]`; byte `k` holds weight `k` (low) and `k+16`
//! (high). It is IQ4_XS without the per-sub-block 6-bit scale — the same
//! codebook, one fp16 scale per 32 — so, like IQ4_XS, it maps onto Q8_0
//! exactly: `d` stays `d`, `q` becomes the codebook value (which spans
//! -127..113 and fits int8). Nothing is even rounded here.

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;
use crate::quant::iq4_xs::KVALUES_IQ4NL;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 18;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ4_NL {
    pub d: u16,
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockIQ4_NL>() == BYTES_PER_BLOCK);

pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);
    let blocks: &[BlockIQ4_NL] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);
    for (b, ob) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d = f16_to_f32(b.d);
        for k in 0..16 {
            ob[k]      = d * KVALUES_IQ4NL[(b.qs[k] & 0x0F) as usize] as f32;
            ob[k + 16] = d * KVALUES_IQ4NL[(b.qs[k] >> 4)   as usize] as f32;
        }
    }
}

/// Transcode IQ4_NL bytes into the equivalent Q8_0 bytes. Bit-exact: the
/// scale is copied and the codebook values already are int8.
pub fn transcode_to_q8_0(bytes: &[u8], n_weights: usize) -> Vec<u8> {
    use crate::quant::q8_0::{BlockQ8_0, BYTES_PER_BLOCK as Q8_BYTES};
    assert_eq!(n_weights % BLOCK_SIZE, 0);
    let n_blocks = n_weights / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);
    let blocks: &[BlockIQ4_NL] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);
    let mut out = vec![0u8; n_blocks * Q8_BYTES];
    let dst: &mut [BlockQ8_0] = bytemuck::cast_slice_mut(&mut out);
    for (b, o) in blocks.iter().zip(dst.iter_mut()) {
        o.d = b.d;
        for k in 0..16 {
            o.qs[k]      = KVALUES_IQ4NL[(b.qs[k] & 0x0F) as usize];
            o.qs[k + 16] = KVALUES_IQ4NL[(b.qs[k] >> 4)   as usize];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    #[test]
    fn transcode_is_bit_exact_against_dequant() {
        const N: usize = 64;
        let mut seed: u64 = 0x1A4A_1000;
        let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                           (seed >> 56) as u8 };
        let mut bytes = vec![0u8; N * BYTES_PER_BLOCK];
        for b in 0..N {
            let off = b * BYTES_PER_BLOCK;
            bytes[off..off + 2].copy_from_slice(
                &f32_to_f16(0.01 * (b as f32 - 30.0)).to_le_bytes());
            for i in 2..BYTES_PER_BLOCK { bytes[off + i] = rng(); }
        }
        let n = N * BLOCK_SIZE;
        let mut direct = vec![0.0f32; n];
        dequantize_to_f32(&bytes, &mut direct);
        let mut via = vec![0.0f32; n];
        crate::quant::q8_0::dequantize_to_f32(&transcode_to_q8_0(&bytes, n), &mut via);
        assert_eq!(direct, via, "iq4_nl -> q8_0 must be bit-exact");
    }
}
