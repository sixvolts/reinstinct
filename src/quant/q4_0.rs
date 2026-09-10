//! Q4_0: 4.5 bpw symmetric legacy quant. Gemma 4's QAT GGUFs are built
//! entirely from it, so it is the only format those files need.
//!
//! 18-byte block / 32 weights laid out as:
//!   fp16  d          block scale
//!   u8    qs[16]     32 nibbles
//!
//! Per-weight: `w = d * (q - 8)`, `q` an unsigned 4-bit nibble. Byte `k`
//! holds weight `k` in its low nibble and weight `k + 16` in its high
//! nibble.
//!
//! No per-sub-block scale plane and no min: one fp16 scale covers all 32
//! weights, and the offset is the constant −8. That makes the matvec
//! cheaper than Q4_K — same nibble traffic, but the kernel skips
//! unpacking a 6-bit (scale, min) pair per sub-block.

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 18;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_0 {
    /// fp16 block scale, raw bits.
    pub d: u16,
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == BYTES_PER_BLOCK);

/// Dequantize whole blocks from `bytes` into `out`. Caller guarantees
/// `out.len() % 32 == 0` and that `bytes` holds that many blocks.
pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);

    let blocks: &[BlockQ4_0] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);

    for (b, out_block) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d = f16_to_f32(b.d);
        for k in 0..16 {
            let lo = (b.qs[k] & 0x0F) as i32 - 8;
            let hi = (b.qs[k] >> 4) as i32 - 8;
            out_block[k]      = d * lo as f32;
            out_block[k + 16] = d * hi as f32;
        }
    }
}

/// Padded sub-block count per repacked row. Same anti-aliasing trick the
/// other repacks use: a power-of-two count gives a power-of-two row
/// stride, which lands every row on one HBM channel and costs ~3× on
/// matvec. One block of padding breaks it; non-power-of-two counts
/// already spread and are left tight.
pub fn repacked_n_sub_padded(in_dim: usize) -> usize {
    let n = in_dim / BLOCK_SIZE;
    if n.is_power_of_two() { n + 1 } else { n }
}

/// Bytes a [`repack_for_matvec`] result occupies for a given shape.
pub fn repacked_len(in_dim: usize, out_dim: usize) -> usize {
    let nsp = repacked_n_sub_padded(in_dim);
    out_dim * nsp * 16 + out_dim * nsp * 2
}

/// Repack a Q4_0 matvec weight `[out_dim, in_dim]` into the two-plane
/// layout `matvec_q4_0_repacked` streams contiguously.
///
///   * **nibble plane** — `out_dim * nsp * 16` bytes. Each 32-weight
///     block gets a contiguous, 16-byte-aligned chunk; `uint32` `j`
///     holds weights `4j..4j+3` in its low nibbles and `16+4j..16+4j+3`
///     in its high nibbles — the form `v_dot4_i32_i8` consumes after one
///     mask/shift, and exactly the order Q4_0 already stores on disk.
///   * **d plane** — `out_dim * nsp * 2` bytes, the fp16 scales pulled
///     into their own stream.
///
/// The nibbles therefore copy across verbatim; the only work is
/// splitting the interleaved `d` out of each 18-byte block so the inner
/// loop reads one aligned `uint4` of quants per `sdot4` group instead of
/// straddling an 18-byte stride.
pub fn repack_for_matvec(bytes: &[u8], in_dim: usize, out_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % BLOCK_SIZE, 0, "Q4_0 in_dim must be a multiple of 32");
    let n_blocks = in_dim / BLOCK_SIZE;
    let nsp      = repacked_n_sub_padded(in_dim);
    let nib_len  = out_dim * nsp * 16;
    let mut out  = vec![0u8; nib_len + out_dim * nsp * 2];

    for row in 0..out_dim {
        for blk in 0..n_blocks {
            let src = (row * n_blocks + blk) * BYTES_PER_BLOCK;
            let dst_nib = (row * nsp + blk) * 16;
            let dst_d   = nib_len + (row * nsp + blk) * 2;
            // on-disk block: u16 d (2 bytes) || 16 nibble bytes
            out[dst_d..dst_d + 2].copy_from_slice(&bytes[src..src + 2]);
            out[dst_nib..dst_nib + 16].copy_from_slice(&bytes[src + 2..src + 18]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    /// `d = 1` makes every dequantized weight the raw `q - 8`, so the
    /// nibble→weight mapping is readable straight off the assertion.
    #[test]
    fn dequant_unit_scale_recovers_nibble_minus_eight() {
        let mut bytes = vec![0u8; BYTES_PER_BLOCK];
        bytes[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // byte k: low nibble = k % 16 (weight k), high nibble = 15 - k%16
        // (weight k+16).
        for k in 0..16 {
            bytes[2 + k] = (k as u8) | ((15 - k as u8) << 4);
        }
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequantize_to_f32(&bytes, &mut out);
        for k in 0..16 {
            assert_eq!(out[k],      k as f32 - 8.0,        "weight {k}");
            assert_eq!(out[k + 16], (15 - k) as f32 - 8.0, "weight {}", k + 16);
        }
    }

    /// Nibble bytes survive the repack untouched and the scales land in
    /// their own plane at the padded stride.
    #[test]
    fn repack_preserves_nibbles_and_splits_scales() {
        let in_dim = 96usize;    // 3 blocks/row — not a power of two, so nsp == 3
        let out_dim = 2usize;
        let n_blocks = in_dim / BLOCK_SIZE;
        assert_eq!(repacked_n_sub_padded(in_dim), n_blocks);

        let mut raw = vec![0u8; out_dim * n_blocks * BYTES_PER_BLOCK];
        for (i, b) in raw.iter_mut().enumerate() { *b = (i % 251) as u8; }
        // Give each block a recognisable scale.
        for blk in 0..out_dim * n_blocks {
            let off = blk * BYTES_PER_BLOCK;
            raw[off..off + 2].copy_from_slice(&f32_to_f16(blk as f32 + 1.0).to_le_bytes());
        }

        let packed = repack_for_matvec(&raw, in_dim, out_dim);
        assert_eq!(packed.len(), repacked_len(in_dim, out_dim));

        let nib_len = out_dim * n_blocks * 16;
        for row in 0..out_dim {
            for blk in 0..n_blocks {
                let src = (row * n_blocks + blk) * BYTES_PER_BLOCK;
                let dn  = (row * n_blocks + blk) * 16;
                assert_eq!(&packed[dn..dn + 16], &raw[src + 2..src + 18],
                           "nibbles row {row} blk {blk}");
                let dd = nib_len + (row * n_blocks + blk) * 2;
                assert_eq!(&packed[dd..dd + 2], &raw[src..src + 2],
                           "scale row {row} blk {blk}");
            }
        }
    }

    /// A power-of-two block count gets one block of padding so the row
    /// stride stops being a power of two.
    #[test]
    fn power_of_two_block_count_is_padded() {
        assert_eq!(repacked_n_sub_padded(32 * 4), 5);
        assert_eq!(repacked_n_sub_padded(32 * 6), 6);
    }

    /// Repacking then reading back through the same index arithmetic the
    /// kernel uses must reproduce the direct dequant bit for bit.
    #[test]
    fn repacked_layout_dequants_to_the_same_weights() {
        let in_dim = 256usize;   // 8 blocks → power of two → nsp = 9
        let out_dim = 3usize;
        let n_blocks = in_dim / BLOCK_SIZE;
        let nsp = repacked_n_sub_padded(in_dim);

        let mut raw = vec![0u8; out_dim * n_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xA5A5_1234;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                           (s >> 56) as u8 };
        for blk in 0..out_dim * n_blocks {
            let off = blk * BYTES_PER_BLOCK;
            raw[off..off + 2].copy_from_slice(
                &f32_to_f16(0.01 * (blk as f32 + 1.0)).to_le_bytes());
            for i in 0..16 { raw[off + 2 + i] = rng(); }
        }

        let mut direct = vec![0.0f32; out_dim * in_dim];
        dequantize_to_f32(&raw, &mut direct);

        let packed = repack_for_matvec(&raw, in_dim, out_dim);
        let nib_len = out_dim * nsp * 16;
        for row in 0..out_dim {
            for blk in 0..n_blocks {
                let dd = nib_len + (row * nsp + blk) * 2;
                let d = f16_to_f32(u16::from_le_bytes([packed[dd], packed[dd + 1]]));
                let dn = (row * nsp + blk) * 16;
                for k in 0..16 {
                    let byte = packed[dn + k];
                    let lo = d * ((byte & 0x0F) as i32 - 8) as f32;
                    let hi = d * ((byte >> 4) as i32 - 8) as f32;
                    let base = row * in_dim + blk * BLOCK_SIZE;
                    assert_eq!(direct[base + k],      lo);
                    assert_eq!(direct[base + k + 16], hi);
                }
            }
        }
    }
}
