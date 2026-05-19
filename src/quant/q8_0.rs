//! Q8_0: 8.5 bpw, fp16 super-scale + 32 signed int8 quants per block.
//! Symmetric: w[i] = d * qs[i].

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 34;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_0 {
    /// fp16 super-scale, raw bits.
    pub d: u16,
    pub qs: [i8; 32],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == BYTES_PER_BLOCK);

/// Dequantize `n_blocks` consecutive blocks from `bytes` into `out`.
/// Caller guarantees `out.len() == n_blocks * BLOCK_SIZE` and
/// `bytes.len() >= n_blocks * BYTES_PER_BLOCK`.
pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);

    let blocks: &[BlockQ8_0] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);

    for (b, out_chunk) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d = f16_to_f32(b.d);
        for (q, o) in b.qs.iter().zip(out_chunk.iter_mut()) {
            *o = d * (*q as f32);
        }
    }
}

/// Padded sub-block count per repacked row. Mirrors the same trick the
/// Q4/5/6_K repacks use: when `in_dim/32` is a power of two every row
/// aliases onto the same HBM channel, costing ~3× on matvec. One
/// padding sub-block breaks that. Non-power-of-two counts already
/// spread across channels and are left tight.
pub fn repacked_n_sub_padded(in_dim: usize) -> usize {
    let n = in_dim / BLOCK_SIZE;
    if n.is_power_of_two() { n + 1 } else { n }
}

/// Bytes a [`repack_for_matvec`] result occupies for a given shape.
pub fn repacked_len(in_dim: usize, out_dim: usize) -> usize {
    let nsp = repacked_n_sub_padded(in_dim);
    out_dim * nsp * 32 + out_dim * nsp * 2
}

/// Repack a Q8_0 matvec weight `[out_dim, in_dim]` into a two-plane
/// layout the `matvec_q8_0_repacked` kernel streams contiguously.
///
///   * **qs plane** — `out_dim * nsp * 32` bytes. Each sub-block's 32
///     int8 quants get a contiguous, 32-byte-aligned chunk (vs the
///     on-disk format's 2-byte-aligned `qs` inside 34-byte blocks).
///     The padded sub-block is zero.
///   * **d plane**  — `out_dim * nsp * 2` bytes. The fp16 d-scales
///     pulled out of the on-disk blocks into their own stream.
///
/// Same total bytes as on-disk modulo padding; the win is alignment
/// (HBM transactions become naturally aligned 32-byte reads, and the
/// dp4a inner loop becomes 1 i32 load per sdot4 instead of 2 uint16
/// loads + an OR-shift).
pub fn repack_for_matvec(bytes: &[u8], in_dim: usize, out_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % BLOCK_SIZE, 0, "Q8_0 in_dim must be a multiple of 32");
    let n_blocks = in_dim / BLOCK_SIZE;
    let nsp      = repacked_n_sub_padded(in_dim);
    let qs_len   = out_dim * nsp * 32;
    let d_len    = out_dim * nsp * 2;
    let mut out  = vec![0u8; qs_len + d_len];

    for row in 0..out_dim {
        for blk in 0..n_blocks {
            let src = (row * n_blocks + blk) * BYTES_PER_BLOCK;
            let dst_qs = (row * nsp + blk) * 32;
            let dst_d  = qs_len + (row * nsp + blk) * 2;
            // on-disk block: u16 d (2 bytes) || i8 qs[32] (32 bytes)
            out[dst_d ..dst_d  + 2 ].copy_from_slice(&bytes[src..src + 2]);
            out[dst_qs..dst_qs + 32].copy_from_slice(&bytes[src + 2..src + 34]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    #[test]
    fn dequant_unit_scale() {
        // Block with d=1.0 should produce qs values directly as f32.
        let mut bytes = vec![0u8; BYTES_PER_BLOCK];
        bytes[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        for i in 0..32 {
            bytes[2 + i] = (i as i8) as u8;
        }
        let mut out = vec![0.0_f32; 32];
        dequantize_to_f32(&bytes, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as f32);
        }
    }

    #[test]
    fn dequant_negative_scale_and_quants() {
        let mut bytes = vec![0u8; BYTES_PER_BLOCK];
        bytes[0..2].copy_from_slice(&f32_to_f16(-0.5).to_le_bytes());
        // qs all = 4 → output all = -0.5 * 4 = -2.0
        for i in 0..32 {
            bytes[2 + i] = 4u8;
        }
        let mut out = vec![0.0_f32; 32];
        dequantize_to_f32(&bytes, &mut out);
        for v in &out {
            assert_eq!(*v, -2.0);
        }
    }

    #[test]
    fn multi_block_dequant() {
        let mut bytes = vec![0u8; 2 * BYTES_PER_BLOCK];
        bytes[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        bytes[BYTES_PER_BLOCK..BYTES_PER_BLOCK + 2]
            .copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        for i in 0..32 {
            bytes[2 + i] = 1;
            bytes[BYTES_PER_BLOCK + 2 + i] = 1;
        }
        let mut out = vec![0.0_f32; 64];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out[0..32].iter().all(|v| *v == 1.0));
        assert!(out[32..64].iter().all(|v| *v == 2.0));
    }
}
