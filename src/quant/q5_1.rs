//! Q5_1: 6 bpw legacy 5-bit, block of 32, asymmetric.
//!
//! 24-byte block / 32 weights:
//!   fp16  d          scale
//!   fp16  m          minimum
//!   u8    qh[4]      the 5th bit of each weight, bit `i` for weight `i`
//!   u8    qs[16]     low nibbles; byte `k` holds weight `k` (low) and `k+16` (high)
//!
//! `w = d * q5 + m` with `q5` in 0..31. Unsloth's UD-XL recipe reaches
//! for it on MoE down experts (the Gemma 4 26B-A4B `UD-Q4_K_XL` has 29
//! of them, 5.3 GB).
//!
//! It runs on the Q5_K kernels: a Q5_K sub-block is also 32 weights of
//! five bits under one `dsc` and one offset `deff` (`w = dsc*q5 - deff`),
//! the difference being only where the scales come from — Q5_K folds
//! 6-bit sub-scales into a super-block `d`/`dmin`, Q5_1 stores fp16 `d`
//! and `m` per block. So the repack is the Q5_K two-plane layout with a
//! third plane of raw fp16 `(d, m)` per sub-block, and the kernels are
//! compiled a second time with `Q5_1_SCALES` (`kernel_source`), which
//! swaps the scale decode for `dsc = d`, `deff = -m`. Exact: nothing is
//! rounded.

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 24;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_1 {
    pub d: u16,
    pub m: u16,
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == BYTES_PER_BLOCK);

pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);
    let blocks: &[BlockQ5_1] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);
    for (b, ob) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d = f16_to_f32(b.d);
        let m = f16_to_f32(b.m);
        let qh = u32::from_le_bytes(b.qh);
        for k in 0..16 {
            let lo = (b.qs[k] & 0x0F) as u32 | (((qh >> k) & 1) << 4);
            let hi = (b.qs[k] >> 4) as u32   | (((qh >> (k + 16)) & 1) << 4);
            ob[k]      = d * lo as f32 + m;
            ob[k + 16] = d * hi as f32 + m;
        }
    }
}

/// The Q5_K kernel sources with the Q5_1 scale decode selected.
pub fn kernel_source(q5k_source: &str) -> String {
    format!("#define Q5_1_SCALES 1\n{q5k_source}")
}

/// Repack a Q5_1 matvec weight `[out_dim, in_dim]` into the Q5_K repacked
/// layout's first two planes plus a per-sub-block fp16 `(d, m)` plane
/// (row stride `nsp = q4_k::repacked_n_sub_padded(in_dim)`):
///   * nibble plane — `nsp*16` B/row: the on-disk `qs` verbatim, already
///     "weight k low / k+16 high", which is the dp4a group order.
///   * qh plane — `nsp*4` B/row: one u32 per sub-block in the Q5_K
///     kernels' order — bit `4g+b` is the 5th bit of the weight in dp4a
///     group `g`, lane `b`, i.e. weight `4j+b` for group `2j` and
///     `16+4j+b` for group `2j+1` (on disk bit `i` is weight `i`).
///   * dm plane — `nsp*4` B/row: `d | m << 16` as raw fp16 bits.
pub fn repack_for_matvec(bytes: &[u8], in_dim: usize, out_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % BLOCK_SIZE, 0, "Q5_1 in_dim must be a multiple of 32");
    let n_sub   = in_dim / BLOCK_SIZE;
    let nsp     = crate::quant::q4_k::repacked_n_sub_padded(in_dim);
    let nib_len = out_dim * nsp * 16;
    let qh_len  = out_dim * nsp * 4;
    let mut out = vec![0u8; nib_len + qh_len + out_dim * nsp * 4];
    let blocks: &[BlockQ5_1] =
        bytemuck::cast_slice(&bytes[..out_dim * n_sub * BYTES_PER_BLOCK]);
    for row in 0..out_dim {
        for sb in 0..n_sub {
            let b = &blocks[row * n_sub + sb];
            let i = row * nsp + sb;
            out[i * 16..i * 16 + 16].copy_from_slice(&b.qs);
            let qh = u32::from_le_bytes(b.qh);
            let mut packed = 0u32;
            for j in 0..4 {
                for bb in 0..4 {
                    packed |= ((qh >> (4 * j + bb)) & 1)      << (8 * j + bb);       // group 2j
                    packed |= ((qh >> (16 + 4 * j + bb)) & 1) << (8 * j + 4 + bb);   // group 2j+1
                }
            }
            out[nib_len + i * 4..nib_len + i * 4 + 4].copy_from_slice(&packed.to_le_bytes());
            let dm = nib_len + qh_len + i * 4;
            out[dm..dm + 2].copy_from_slice(&b.d.to_le_bytes());
            out[dm + 2..dm + 4].copy_from_slice(&b.m.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    /// Reading the planes back the way the kernels do (nibble k low /
    /// k+16 high, qh bit 4g+b, `dsc = d`, `deff = -m`) reproduces the
    /// direct dequant bit for bit.
    #[test]
    fn repack_is_bit_exact_against_dequant() {
        let in_dim = 256usize;   // 8 sub-blocks -> pow2 -> nsp = 9
        let out_dim = 5usize;
        let n_sub = in_dim / BLOCK_SIZE;
        let nsp = crate::quant::q4_k::repacked_n_sub_padded(in_dim);
        assert_eq!(nsp, 9);
        let mut seed: u64 = 0x0510_0001;
        let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                           (seed >> 56) as u8 };
        let mut bytes = vec![0u8; out_dim * n_sub * BYTES_PER_BLOCK];
        for b in 0..out_dim * n_sub {
            let off = b * BYTES_PER_BLOCK;
            bytes[off..off + 2].copy_from_slice(&f32_to_f16(0.01 * (b as f32 + 1.0)).to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&f32_to_f16(-0.1 * (b as f32 % 7.0)).to_le_bytes());
            for i in 4..BYTES_PER_BLOCK { bytes[off + i] = rng(); }
        }
        let mut direct = vec![0.0f32; out_dim * in_dim];
        dequantize_to_f32(&bytes, &mut direct);

        let packed = repack_for_matvec(&bytes, in_dim, out_dim);
        let (nib_len, qh_len) = (out_dim * nsp * 16, out_dim * nsp * 4);
        assert_eq!(packed.len(), nib_len + qh_len + out_dim * nsp * 4);
        for row in 0..out_dim {
            for sb in 0..n_sub {
                let i = row * nsp + sb;
                let nib = &packed[i * 16..i * 16 + 16];
                let qh = u32::from_le_bytes(packed[nib_len + i * 4..nib_len + i * 4 + 4].try_into().unwrap());
                let dm = nib_len + qh_len + i * 4;
                let dsc = f16_to_f32(u16::from_le_bytes([packed[dm], packed[dm + 1]]));
                let deff = -f16_to_f32(u16::from_le_bytes([packed[dm + 2], packed[dm + 3]]));
                for w in 0..32 {
                    let (g, b) = (w / 4, w % 4);
                    // dp4a group g = word j (g/2), low nibbles for even g.
                    let byte = nib[(g / 2) * 4 + b];
                    let n = if g % 2 == 0 { byte & 0x0F } else { byte >> 4 } as u32;
                    let q5 = n | (((qh >> (4 * g + b)) & 1) << 4);
                    let widx = if g % 2 == 0 { (g / 2) * 4 + b } else { 16 + (g / 2) * 4 + b };
                    assert_eq!(dsc * q5 as f32 - deff, direct[row * in_dim + sb * 32 + widx],
                               "row {row} sb {sb} w {widx}");
                }
            }
        }
        assert!(kernel_source("").starts_with("#define Q5_1_SCALES 1\n"));
    }
}
