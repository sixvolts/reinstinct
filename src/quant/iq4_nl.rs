//! IQ4_NL: 4.5 bpw non-linear 4-bit, block of 32.
//!
//! 18-byte block / 32 weights:
//!   fp16  d          block scale
//!   u8    qs[16]     32 nibbles indexing the 16-entry `KVALUES_IQ4NL` codebook
//!
//! `w = d * KVALUES[nibble]`; byte `k` holds weight `k` (low) and `k+16`
//! (high). It is IQ4_XS without the per-sub-block 6-bit scale — the same
//! codebook, one fp16 scale per 32. That makes it a *subset* of the
//! repacked IQ4_XS layout (`iq4_xs::repack_for_matvec`: a nibble plane
//! and one fp16 scale per 32-weight sub-block): the nibbles copy across
//! verbatim and `d` is the sub-block scale as-is. Nothing is rounded, and
//! the tensor runs on the IQ4_XS kernels tagged as `IQ4_XS`.

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

/// Repack an IQ4_NL matvec weight `[out_dim, in_dim]` into the repacked
/// IQ4_XS layout, bit for bit: the 16 nibble bytes per block copy across
/// and `d` is the per-sub-block fp16 scale. The result is consumed by the
/// IQ4_XS kernels, so callers tag it `GgmlType::IQ4_XS`.
pub fn repack_for_matvec(bytes: &[u8], in_dim: usize, out_dim: usize) -> Vec<u8> {
    use crate::quant::iq4_xs::repacked_n_sub_padded;
    assert_eq!(in_dim % BLOCK_SIZE, 0, "IQ4_NL in_dim must be a multiple of 32");
    let n_sub   = in_dim / BLOCK_SIZE;
    let nsp     = repacked_n_sub_padded(in_dim);
    let nib_len = out_dim * nsp * 16;
    let mut out = vec![0u8; nib_len + out_dim * nsp * 2];
    let blocks: &[BlockIQ4_NL] =
        bytemuck::cast_slice(&bytes[..out_dim * n_sub * BYTES_PER_BLOCK]);
    for row in 0..out_dim {
        for sb in 0..n_sub {
            let b = &blocks[row * n_sub + sb];
            let dst_nib = (row * nsp + sb) * 16;
            out[dst_nib..dst_nib + 16].copy_from_slice(&b.qs);
            let dst_d = nib_len + (row * nsp + sb) * 2;
            out[dst_d..dst_d + 2].copy_from_slice(&b.d.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    /// Reading the repacked planes back through the IQ4_XS kernels' index
    /// arithmetic (nibble `k` low / `k+16` high, codebook, fp16 scale)
    /// must reproduce the direct dequant exactly — nothing is rounded.
    #[test]
    fn repack_is_bit_exact_against_dequant() {
        use crate::quant::iq4_xs::repacked_n_sub_padded;
        let in_dim = 128usize;     // 4 sub-blocks -> pow2 -> nsp = 5
        let out_dim = 3usize;
        let n_sub = in_dim / BLOCK_SIZE;
        let nsp = repacked_n_sub_padded(in_dim);
        assert_eq!(nsp, 5);
        let mut seed: u64 = 0x1A4A_1000;
        let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                           (seed >> 56) as u8 };
        let mut bytes = vec![0u8; out_dim * n_sub * BYTES_PER_BLOCK];
        for b in 0..out_dim * n_sub {
            let off = b * BYTES_PER_BLOCK;
            bytes[off..off + 2].copy_from_slice(
                &f32_to_f16(0.01 * (b as f32 - 5.0)).to_le_bytes());
            for i in 2..BYTES_PER_BLOCK { bytes[off + i] = rng(); }
        }
        let mut direct = vec![0.0f32; out_dim * in_dim];
        dequantize_to_f32(&bytes, &mut direct);

        let packed = repack_for_matvec(&bytes, in_dim, out_dim);
        let nib_len = out_dim * nsp * 16;
        assert_eq!(packed.len(), nib_len + out_dim * nsp * 2);
        for row in 0..out_dim {
            for sb in 0..n_sub {
                let nib = &packed[(row * nsp + sb) * 16..][..16];
                let d_off = nib_len + (row * nsp + sb) * 2;
                let d = f16_to_f32(u16::from_le_bytes([packed[d_off], packed[d_off + 1]]));
                for k in 0..16 {
                    let lo = d * KVALUES_IQ4NL[(nib[k] & 0x0F) as usize] as f32;
                    let hi = d * KVALUES_IQ4NL[(nib[k] >> 4) as usize] as f32;
                    assert_eq!(lo, direct[row * in_dim + sb * 32 + k]);
                    assert_eq!(hi, direct[row * in_dim + sb * 32 + k + 16]);
                }
            }
        }
    }
}
