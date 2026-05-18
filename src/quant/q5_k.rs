//! Q5_K: 5.5 bpw — Q4_K plus a 32-byte high-bit array making each quant 5 bits.
//!
//! 176-byte block / 256 weights:
//!   fp16  d            super-block scale
//!   fp16  dmin         super-block min
//!   u8    scales[12]   identical packing to Q4_K (get_scale_min_k4)
//!   u8    qh[32]       one high-bit per weight, transposed across chunks
//!   u8    qs[128]      low 4 bits per weight (same layout as Q4_K's qs)
//!
//! Per-weight: `q5 = (qs_nibble) | ((qh_bit) << 4)` ∈ 0..31
//! `w = d * sub_d * q5 - dmin * sub_m`
//!
//! qh transpose: the 8 bits of qh[l] feed weight positions `chunk*64 + sub*32 + l`,
//! where `chunk ∈ 0..4` selects two adjacent bits (`chunk*2`, `chunk*2 + 1`)
//! within the byte.

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 176;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ5_K>() == BYTES_PER_BLOCK);

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
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

    let blocks: &[BlockQ5_K] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);

    for (b, out_block) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d    = f16_to_f32(b.d);
        let dmin = f16_to_f32(b.dmin);

        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for chunk in 0..4 {
            let q_off = chunk * 32;
            let y_off = chunk * 64;
            let sub_a = chunk * 2;
            let sub_b = chunk * 2 + 1;

            let (sc_a, m_a) = get_scale_min_k4(sub_a, &b.scales);
            let (sc_b, m_b) = get_scale_min_k4(sub_b, &b.scales);
            let d1 = d * sc_a as f32;
            let m1 = dmin * m_a as f32;
            let d2 = d * sc_b as f32;
            let m2 = dmin * m_b as f32;

            let qs = &b.qs[q_off..q_off + 32];
            for l in 0..32 {
                let q_lo = (qs[l] & 0x0F) as i32 + if b.qh[l] & u1 != 0 { 16 } else { 0 };
                let q_hi = (qs[l] >> 4)   as i32 + if b.qh[l] & u2 != 0 { 16 } else { 0 };
                out_block[y_off + l]      = d1 * q_lo as f32 - m1;
                out_block[y_off + l + 32] = d2 * q_hi as f32 - m2;
            }

            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Repack a Q5_K matvec weight `[out_dim, in_dim]` into three contiguous
/// planes (row stride `nsp = q4_k::repacked_n_sub_padded(in_dim)`):
///   * nibble plane — `nsp*16` B/row, low 4 bits per weight, permuted as
///     in `q4_k::repack_for_matvec`.
///   * qh plane — `nsp*4` B/row. Per sub-block one u32: bit `4g+b` is the
///     5th bit of the `b`-th weight of dp4a group `g`.
///   * scale plane — `nsp*2` B/row: the 6-bit `sc`, `m` as two u8.
///   * superblock plane — `(in_dim/256)*4` B/row: raw fp16 `d`, `dmin`.
/// dsc = d·sc, deff = dmin·m are formed on the fly (format v2).
pub fn repack_for_matvec(bytes: &[u8], in_dim: usize, out_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % BLOCK_SIZE, 0, "Q5_K in_dim must be a multiple of 256");
    let n_blocks = in_dim / BLOCK_SIZE;
    let nsp      = crate::quant::q4_k::repacked_n_sub_padded(in_dim);
    let nib_len  = out_dim * nsp * 16;
    let qh_len   = out_dim * nsp * 4;
    let sm_len   = out_dim * nsp * 2;
    let mut out  = vec![0u8; nib_len + qh_len + sm_len + out_dim * n_blocks * 4];

    let blocks: &[BlockQ5_K] =
        bytemuck::cast_slice(&bytes[..out_dim * n_blocks * BYTES_PER_BLOCK]);

    for row in 0..out_dim {
        for blk in 0..n_blocks {
            let b = &blocks[row * n_blocks + blk];

            // Superblock plane: raw fp16 d, dmin once per 256 weights.
            let dd_off = nib_len + qh_len + sm_len + (row * n_blocks + blk) * 4;
            out[dd_off..dd_off + 2].copy_from_slice(&b.d.to_le_bytes());
            out[dd_off + 2..dd_off + 4].copy_from_slice(&b.dmin.to_le_bytes());

            for s in 0..8usize {
                let gsb = blk * 8 + s;
                let qs  = &b.qs[(s >> 1) * 32..(s >> 1) * 32 + 32];

                let mut w = [0u8; 32];
                let mut hb = [0u8; 32];
                for l in 0..32 {
                    w[l]  = if s & 1 == 0 { qs[l] & 0x0F } else { qs[l] >> 4 };
                    hb[l] = (b.qh[l] >> s) & 1;
                }

                let nib_off = (row * nsp + gsb) * 16;
                let mut qh_packed: u32 = 0;
                for j in 0..4 {
                    for bb in 0..4 {
                        out[nib_off + j * 4 + bb] =
                            w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                        // dp4a group 2j → weights 4j+bb, group 2j+1 → 16+4j+bb.
                        qh_packed |= (hb[4 * j + bb] as u32)      << (4 * (2 * j)     + bb);
                        qh_packed |= (hb[16 + 4 * j + bb] as u32) << (4 * (2 * j + 1) + bb);
                    }
                }
                let qh_off = nib_len + (row * nsp + gsb) * 4;
                out[qh_off..qh_off + 4].copy_from_slice(&qh_packed.to_le_bytes());

                let (sc, m) = get_scale_min_k4(s, &b.scales);
                let sm_off = nib_len + qh_len + (row * nsp + gsb) * 2;
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
    use crate::quant::half::f32_to_f16;
    use crate::quant::q4_k;

    fn synth_block(d: f32, dmin: f32, sc: u8, m: u8, nibble: u8, high_bit: bool) -> Vec<u8> {
        assert!(sc < 64 && m < 64 && nibble < 16);
        let mut bytes = vec![0u8; BYTES_PER_BLOCK];
        bytes[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        bytes[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());

        let sc_hi2 = (sc >> 4) & 0x3;
        let m_hi2  = (m  >> 4) & 0x3;
        let sc_lo4 = sc & 0x0F;
        let m_lo4  = m  & 0x0F;
        for j in 0..4 {
            bytes[4 + j]  = (sc & 0x3F) | (sc_hi2 << 6);
            bytes[8 + j]  = (m  & 0x3F) | (m_hi2  << 6);
            bytes[12 + j] = sc_lo4 | (m_lo4 << 4);
        }

        // qh: 32 bytes starting at offset 16; if high_bit set, fill with 0xFF (every weight has bit 4 set).
        let qh_byte = if high_bit { 0xFFu8 } else { 0 };
        for i in 0..32 {
            bytes[16 + i] = qh_byte;
        }

        // qs: 128 bytes starting at offset 48 (16 + 32).
        let qs_byte = (nibble & 0x0F) | ((nibble & 0x0F) << 4);
        for i in 0..128 {
            bytes[48 + i] = qs_byte;
        }
        bytes
    }

    #[test]
    fn q5k_with_zero_high_bit_matches_q4k() {
        // Same fields but no high bit → q5_k output identical to q4_k output.
        let q5_bytes = synth_block(2.0, 1.0, 3, 4, 5, false);
        let mut q5_out = vec![0.0_f32; 256];
        dequantize_to_f32(&q5_bytes, &mut q5_out);

        let mut q4_bytes = vec![0u8; q4_k::BYTES_PER_BLOCK];
        q4_bytes[0..2].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        q4_bytes[2..4].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        for j in 0..4 {
            q4_bytes[4 + j]  = (3 & 0x3F) | (((3 >> 4) & 0x3) << 6);
            q4_bytes[8 + j]  = (4 & 0x3F) | (((4 >> 4) & 0x3) << 6);
            q4_bytes[12 + j] = (3 & 0x0F) | ((4 & 0x0F) << 4);
        }
        for i in 0..128 {
            q4_bytes[16 + i] = 5 | (5 << 4);
        }
        let mut q4_out = vec![0.0_f32; 256];
        q4_k::dequantize_to_f32(&q4_bytes, &mut q4_out);

        assert_eq!(q5_out, q4_out);
    }

    #[test]
    fn high_bit_adds_16_to_quant() {
        // d=1, dmin=0, sc=1, m=0, nibble=2, high_bit=true → q5 = 2 + 16 = 18, w = 18.
        let bytes = synth_block(1.0, 0.0, 1, 0, 2, true);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == 18.0));
    }

    #[test]
    fn qh_bit_position_per_chunk() {
        // qh[0] = 0b0000_0001 → only chunk 0 sub-block 0 (the very first bit, u1=1)
        // gets the high bit. All other weights share the same low nibble (= 0)
        // so they're 0; weight 0 is 0 + 16 = 16.
        let mut bytes = synth_block(1.0, 0.0, 1, 0, 0, false);
        bytes[16] = 0b0000_0001;
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert_eq!(out[0], 16.0);
        assert!(out[1..].iter().all(|v| *v == 0.0));

        // qh[31] = 0b1000_0000 → only chunk 3 sub-block 1 (u2 bit at position 7),
        // weight at position 3*64 + 32 + 31 = 255.
        let mut bytes = synth_block(1.0, 0.0, 1, 0, 0, false);
        bytes[16 + 31] = 0b1000_0000;
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert_eq!(out[255], 16.0);
        assert!(out[..255].iter().all(|v| *v == 0.0));
    }
}
