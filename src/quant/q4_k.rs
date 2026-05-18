//! Q4_K: 4.5 bpw asymmetric K-quant.
//!
//! 144-byte block / 256 weights laid out as:
//!   fp16  d            super-block scale
//!   fp16  dmin         super-block min
//!   u8    scales[12]   8 sub-block 6-bit scales + 8 6-bit mins, bit-packed
//!   u8    qs[128]      256 nibbles (low/high pairs feed adjacent sub-blocks)
//!
//! Per-weight: `w = d * sub_d * q - dmin * sub_m`
//! where `q` is a 4-bit unsigned nibble and (sub_d, sub_m) are the per-sub-block
//! 6-bit values unpacked via `get_scale_min_k4`.

use bytemuck::{Pod, Zeroable};

use crate::quant::half::{f16_to_f32, f32_to_f16};

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 144;
pub const N_SUBBLOCKS: usize = 8;
pub const SUBBLOCK_SIZE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_K {
    pub d: u16,         // fp16 bits
    pub dmin: u16,      // fp16 bits
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_K>() == BYTES_PER_BLOCK);

/// ggml's `get_scale_min_k4` — unpack the 6-bit (scale, min) pair for sub-block `j`.
///
/// Layout of the 12-byte scales array:
///   bytes 0..3:  low 6 bits = sub_scale[0..3];  high 2 bits → high-nibble of sub_scale[4..7]
///   bytes 4..7:  low 6 bits = sub_min[0..3];    high 2 bits → high-nibble of sub_min[4..7]
///   bytes 8..11: low 4 bits = low-nibble of sub_scale[4..7]; high 4 bits = low-nibble of sub_min[4..7]
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
        (d, m)
    }
}

pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);

    let blocks: &[BlockQ4_K] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);

    for (b, out_block) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d    = f16_to_f32(b.d);
        let dmin = f16_to_f32(b.dmin);

        // Walk 4 chunks of 64 weights (2 sub-blocks per chunk).
        for chunk in 0..4 {
            let q_off  = chunk * 32;
            let y_off  = chunk * 64;
            let sub_a  = chunk * 2;
            let sub_b  = chunk * 2 + 1;

            let (sc_a, m_a) = get_scale_min_k4(sub_a, &b.scales);
            let (sc_b, m_b) = get_scale_min_k4(sub_b, &b.scales);

            let d1 = d * sc_a as f32;
            let m1 = dmin * m_a as f32;
            let d2 = d * sc_b as f32;
            let m2 = dmin * m_b as f32;

            let qs = &b.qs[q_off..q_off + 32];
            for l in 0..32 {
                out_block[y_off + l]      = d1 * (qs[l] & 0x0F) as f32 - m1;
                out_block[y_off + l + 32] = d2 * (qs[l] >> 4)   as f32 - m2;
            }
        }
    }
}

/// Sub-blocks per repacked row. When the natural count `in_dim/32` is a
/// power of two the row stride (`n_sub * 16` bytes) is also a power of
/// two, so every row aliases onto the same HBM channel — a ~3× matvec
/// slowdown. One sub-block of padding breaks that. A non-power-of-two
/// count already spreads across channels, so it is left tight (padding
/// it would cost ~6% for no benefit).
pub fn repacked_n_sub_padded(in_dim: usize) -> usize {
    let n_sub = in_dim / SUBBLOCK_SIZE;
    if n_sub.is_power_of_two() { n_sub + 1 } else { n_sub }
}

/// Bytes a [`repack_for_matvec`] result occupies for a given matvec shape.
pub fn repacked_len(in_dim: usize, out_dim: usize) -> usize {
    let nsp = repacked_n_sub_padded(in_dim);
    let n_blocks = in_dim / BLOCK_SIZE;
    out_dim * nsp * 16 + out_dim * nsp * 2 + out_dim * n_blocks * 4
}

/// Repack a Q4_K matvec weight `[out_dim, in_dim]` into the three-plane
/// layout (format v2) the GPU `matvec_q4k_repacked` kernel streams
/// contiguously. v2 stores the scales at native Q4_K density (the v1
/// fp16-expanded scale plane was ~11% larger than the on-disk format,
/// and decode is HBM-bound).
///
///   * **nibble plane** — `out_dim * nsp * 16` bytes, `nsp =
///     repacked_n_sub_padded(in_dim)` sub-blocks/row. Each 32-weight
///     sub-block gets a contiguous 16 bytes; `uint32` `j` holds weights
///     `4j..4j+3` (low nibbles) and `16+4j..16+4j+3` (high nibbles) —
///     the form `v_dot4_i32_i8` consumes directly.
///   * **scale plane** — `out_dim * nsp * 2` bytes. Per sub-block, the
///     6-bit `sc` then `m` as two `u8` (the `get_scale_min_k4` output).
///   * **superblock plane** — `out_dim * (in_dim/256) * 4` bytes. Per
///     256-weight superblock, the raw `fp16` `d` then `dmin`.
///
/// The kernel forms `dsc = d·sc` and `deff = dmin·m` on the fly — two
/// cheap multiplies, vs ~10% more HBM traffic for the v1 pre-multiplied
/// fp16 plane.
///
/// `in_dim` must be a multiple of 256; `bytes` must hold at least
/// `out_dim * (in_dim/256)` Q4_K blocks.
pub fn repack_for_matvec(bytes: &[u8], in_dim: usize, out_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % BLOCK_SIZE, 0, "Q4_K in_dim must be a multiple of 256");
    let n_blocks = in_dim / BLOCK_SIZE;
    let nsp      = repacked_n_sub_padded(in_dim);   // padded sub-blocks per row
    let nib_len  = out_dim * nsp * 16;
    let sm_len   = out_dim * nsp * 2;
    let mut out  = vec![0u8; nib_len + sm_len + out_dim * n_blocks * 4];

    let blocks: &[BlockQ4_K] =
        bytemuck::cast_slice(&bytes[..out_dim * n_blocks * BYTES_PER_BLOCK]);

    for row in 0..out_dim {
        for blk in 0..n_blocks {
            let b = &blocks[row * n_blocks + blk];

            // Superblock plane: raw fp16 d, dmin once per 256 weights.
            let dd_off = nib_len + sm_len + (row * n_blocks + blk) * 4;
            out[dd_off..dd_off + 2].copy_from_slice(&b.d.to_le_bytes());
            out[dd_off + 2..dd_off + 4].copy_from_slice(&b.dmin.to_le_bytes());

            for s in 0..N_SUBBLOCKS {
                let gsb = blk * N_SUBBLOCKS + s;    // sub-block index within the row

                // This sub-block's 32 nibble weights.
                let qs = &b.qs[(s >> 1) * 32..(s >> 1) * 32 + 32];
                let mut w = [0u8; 32];
                if s & 1 == 0 {
                    for k in 0..32 { w[k] = qs[k] & 0x0F; }
                } else {
                    for k in 0..32 { w[k] = qs[k] >> 4; }
                }

                // Pack: nibble byte 4j+b = w[4j+b] | (w[16+4j+b] << 4).
                let nib_off = (row * nsp + gsb) * 16;
                for j in 0..4 {
                    for bb in 0..4 {
                        out[nib_off + j * 4 + bb] =
                            w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                    }
                }

                // Scale plane: the 6-bit sc, m as two u8.
                let (sc, m) = get_scale_min_k4(s, &b.scales);
                let sm_off = nib_len + (row * nsp + gsb) * 2;
                out[sm_off]     = sc;
                out[sm_off + 1] = m;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::{f16_to_f32, f32_to_f16};

    /// Build a Q4_K block with all 8 sub-blocks sharing the same (sc, m) value
    /// and all 256 nibbles set to `nibble`.
    fn synth_block(d: f32, dmin: f32, sc: u8, m: u8, nibble: u8) -> Vec<u8> {
        assert!(sc < 64 && m < 64 && nibble < 16);
        let mut bytes = vec![0u8; BYTES_PER_BLOCK];
        bytes[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        bytes[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());

        // The 12-byte scales array starts at byte offset 4 (after d + dmin).
        // For all 8 sub-blocks sharing the same (sc, m), the inverse of
        // get_scale_min_k4 packs as:
        //   scales[0..3] = (sc & 0x3F) | (sc_hi2 << 6)   sc_hi2 = (sc >> 4) & 0x3
        //   scales[4..7] = (m  & 0x3F) | (m_hi2  << 6)   m_hi2  = (m  >> 4) & 0x3
        //   scales[8..11] = (sc & 0xF) | ((m & 0xF) << 4)
        let sc_hi2 = (sc >> 4) & 0x3;
        let m_hi2  = (m  >> 4) & 0x3;
        let sc_lo4 = sc & 0x0F;
        let m_lo4  = m  & 0x0F;
        for j in 0..4 {
            bytes[4 + j]  = (sc & 0x3F) | (sc_hi2 << 6);
            bytes[8 + j]  = (m  & 0x3F) | (m_hi2  << 6);
            bytes[12 + j] = sc_lo4 | (m_lo4 << 4);
        }

        let qs_byte = (nibble & 0x0F) | ((nibble & 0x0F) << 4);
        for i in 0..128 {
            bytes[16 + i] = qs_byte;
        }
        bytes
    }

    #[test]
    fn unit_scale_zero_min_extracts_nibbles() {
        // d=1, dmin=0, sc=1, m=0, nibble=k → output all = k.
        for k in [0u8, 1, 7, 15] {
            let bytes = synth_block(1.0, 0.0, 1, 0, k);
            let mut out = vec![0.0_f32; 256];
            dequantize_to_f32(&bytes, &mut out);
            assert!(out.iter().all(|v| *v == k as f32),
                "k={k}: got first values {:?}", &out[..8]);
        }
    }

    #[test]
    fn asymmetric_formula() {
        // d=2, dmin=1, sc=3, m=4, nibble=5 → 2*3*5 - 1*4 = 26
        let bytes = synth_block(2.0, 1.0, 3, 4, 5);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == 26.0));
    }

    #[test]
    fn sub_scale_high_bits_packed_correctly() {
        // Use sc=63 (all 6 bits set, exercises both low-4 and high-2 packing
        // for sub-blocks 4..7) and m=63 (same). nibble=1 → 1 * 63 * 1 - 1 * 63 = 0.
        let bytes = synth_block(1.0, 1.0, 63, 63, 1);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == 0.0));

        // Different nibble in different sub-blocks would distinguish sub-blocks
        // 0..3 (q[0..3]) from 4..7 (q[8..11] + high bits from q[0..7]).
        // Use sc=63 dmin=0 nibble=2 → 1 * 63 * 2 - 0 = 126.
        let bytes = synth_block(1.0, 0.0, 63, 0, 2);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == 126.0));
    }

    #[test]
    fn repack_matches_reference_dequant() {
        // Two blocks, varied nibbles — exercises the per-sub-block nibble
        // permutation and the pre-multiplied fp16 scales. d=2, sc=3 and
        // dmin=1, m=4 keep d·sc=6 and dmin·m=4 exactly fp16-representable
        // so the comparison can be tight.
        let n_blocks = 2usize;
        let mut raw = Vec::new();
        for blk in 0..n_blocks {
            let mut b = synth_block(2.0, 1.0, 3, 4, 0);
            for i in 0..128 {
                b[16 + i] = ((blk * 128 + i) * 5 + 1) as u8;
            }
            raw.extend_from_slice(&b);
        }
        let in_dim = n_blocks * 256;
        let out_dim = 1;

        let mut reference = vec![0.0f32; in_dim];
        dequantize_to_f32(&raw, &mut reference);

        let packed = repack_for_matvec(&raw, in_dim, out_dim);
        let n_sub = in_dim / SUBBLOCK_SIZE;
        let nib_len = out_dim * repacked_n_sub_padded(in_dim) * 16;
        let sm_len  = out_dim * repacked_n_sub_padded(in_dim) * 2;

        // Decode the v2 repacked layout exactly as the GPU kernel does.
        // out_dim == 1 here, so the row stride drops out: sub-block gsb
        // sits at offset gsb in the nibble/scale planes and superblock
        // gsb/8 in the d/dmin plane.
        let mut got = vec![0.0f32; in_dim];
        for gsb in 0..n_sub {
            let nib = &packed[gsb * 16..gsb * 16 + 16];
            let smo = nib_len + gsb * 2;
            let sc = packed[smo] as f32;
            let m  = packed[smo + 1] as f32;
            let ddo = nib_len + sm_len + (gsb / N_SUBBLOCKS) * 4;
            let d    = f16_to_f32(u16::from_le_bytes([packed[ddo], packed[ddo + 1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([packed[ddo + 2], packed[ddo + 3]]));
            let dsc  = d * sc;
            let deff = dmin * m;
            let (blk, s) = (gsb / N_SUBBLOCKS, gsb % N_SUBBLOCKS);
            let base = blk * 256 + (s / 2) * 64 + (s & 1) * 32;
            for k in 0..32 {
                let nibble = if k < 16 { nib[k] & 0x0F } else { nib[k - 16] >> 4 };
                got[base + k] = dsc * nibble as f32 - deff;
            }
        }
        for i in 0..in_dim {
            assert!((got[i] - reference[i]).abs() < 1e-3,
                "mismatch at {i}: got {} ref {}", got[i], reference[i]);
        }
    }

    #[test]
    fn nibble_low_high_split_writes_to_adjacent_sub_blocks() {
        // Different low and high nibbles per qs byte → first 32 outputs use
        // low nibble (sub-block 0), next 32 use high nibble (sub-block 1).
        // d=1, dmin=0, all sub-scales=1.
        let mut bytes = synth_block(1.0, 0.0, 1, 0, 0);
        // Override qs with low=3, high=12.
        for i in 0..128 {
            bytes[16 + i] = 3 | (12 << 4);
        }
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        // Per chunk (64 weights): first 32 = 3, next 32 = 12.
        for chunk in 0..4 {
            let off = chunk * 64;
            assert!(out[off..off + 32].iter().all(|v| *v == 3.0));
            assert!(out[off + 32..off + 64].iter().all(|v| *v == 12.0));
        }
    }
}
