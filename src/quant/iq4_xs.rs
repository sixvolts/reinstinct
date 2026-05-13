//! IQ4_XS: 4.25 bpw importance-quantized 4-bit, super-block of 256 weights.
//!
//! 136-byte block:
//!   fp16  d              super-block scale
//!   u16   scales_h       high 2 bits of each of 8 sub-block scales (8 × 2 = 16)
//!   u8    scales_l[4]    low 4 bits of each scale (8 × 4 = 32 bits)
//!   u8    qs[128]        256 nibbles indexing kvalues_iq4nl[16]
//!
//! Per sub-block (32 weights, 16 qs bytes):
//!   ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xF | ((scales_h >> 2*ib) & 0x3) << 4   ∈ 0..63
//!   dl = d * (ls - 32)                                                            scale ∈ [-32, 31]
//!   For l in 0..16:
//!     w[l]      = dl * LUT[qs[l] & 0xF]
//!     w[l + 16] = dl * LUT[qs[l] >> 4]
//!
//! Note: nibble layout differs from Q4_K — here low/high nibbles fill adjacent
//! halves of the SAME sub-block, not adjacent sub-blocks.

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 136;

/// Non-uniform 4-bit codebook shared by IQ4_NL and IQ4_XS.
/// Designed to match a roughly-Gaussian weight distribution.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ4_XS {
    pub d: u16,
    pub scales_h: u16,
    pub scales_l: [u8; 4],
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockIQ4_XS>() == BYTES_PER_BLOCK);

pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);

    let blocks: &[BlockIQ4_XS] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);

    for (b, out_block) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d = f16_to_f32(b.d);

        for ib in 0..8usize {
            let ls_lo = (b.scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            let ls_hi = ((b.scales_h >> (2 * ib)) & 0x3) as u8;
            let ls = (ls_lo | (ls_hi << 4)) as i32; // 0..63
            let dl = d * (ls - 32) as f32;

            let qs_off = ib * 16;
            let y_off  = ib * 32;
            for l in 0..16 {
                let lo = b.qs[qs_off + l] & 0x0F;
                let hi = b.qs[qs_off + l] >> 4;
                out_block[y_off + l]      = dl * KVALUES_IQ4NL[lo as usize] as f32;
                out_block[y_off + l + 16] = dl * KVALUES_IQ4NL[hi as usize] as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    /// All sub-block scales = `ls_value` (must fit in 6 bits, 0..63).
    /// All nibbles = `nibble`.
    fn synth_block(d: f32, ls_value: u8, nibble: u8) -> Vec<u8> {
        assert!(ls_value < 64 && nibble < 16);
        let mut bytes = vec![0u8; BYTES_PER_BLOCK];

        bytes[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());

        // scales_h: 16 bits, 2 bits per sub-block × 8 sub-blocks.
        let hi2 = (ls_value >> 4) & 0x3;
        let mut scales_h: u16 = 0;
        for ib in 0..8 {
            scales_h |= (hi2 as u16) << (2 * ib);
        }
        bytes[2..4].copy_from_slice(&scales_h.to_le_bytes());

        // scales_l: 4 bytes, 4 bits per sub-block × 8 sub-blocks.
        let lo4 = ls_value & 0x0F;
        let scales_l_byte = lo4 | (lo4 << 4);
        for i in 0..4 {
            bytes[4 + i] = scales_l_byte;
        }

        // qs: 128 bytes, all nibbles = `nibble`.
        let qs_byte = (nibble & 0x0F) | ((nibble & 0x0F) << 4);
        for i in 0..128 {
            bytes[8 + i] = qs_byte;
        }
        bytes
    }

    #[test]
    fn zero_scale_when_ls_equals_32() {
        // ls=32 → (ls - 32) = 0 → all outputs zero regardless of d, nibble.
        let bytes = synth_block(7.0, 32, 5);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn lut_lookup_at_each_nibble() {
        // d=1, ls=33 → (ls-32)=1. Output = LUT[nibble].
        for nibble in 0..16u8 {
            let bytes = synth_block(1.0, 33, nibble);
            let mut out = vec![0.0_f32; 256];
            dequantize_to_f32(&bytes, &mut out);
            let expected = KVALUES_IQ4NL[nibble as usize] as f32;
            assert!(out.iter().all(|v| *v == expected),
                "nibble={nibble}: expected {expected}, first values {:?}", &out[..4]);
        }
    }

    #[test]
    fn negative_scale_via_ls_below_32() {
        // d=1, ls=20 → (ls-32) = -12. Output = -12 * LUT[nibble].
        // Pick nibble=8 → LUT[8] = 1, so output = -12.
        let bytes = synth_block(1.0, 20, 8);
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == -12.0));
    }

    #[test]
    fn nibble_low_high_split_within_subblock() {
        // Force qs to have low=2 (LUT[2]=-83), high=10 (LUT[10]=25).
        // Within each 32-weight sub-block: first 16 = -83 * dl, next 16 = 25 * dl.
        // d=1, ls=33 → dl=1.
        let mut bytes = synth_block(1.0, 33, 0);
        for i in 0..128 {
            bytes[8 + i] = 2 | (10 << 4);
        }
        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        let lo = KVALUES_IQ4NL[2] as f32;
        let hi = KVALUES_IQ4NL[10] as f32;
        for sb in 0..8 {
            let off = sb * 32;
            assert!(out[off..off + 16].iter().all(|v| *v == lo));
            assert!(out[off + 16..off + 32].iter().all(|v| *v == hi));
        }
    }

    #[test]
    fn per_subblock_scale_distinguishes_outputs() {
        // Set sub-block scale ls = 33 + sb (so dl values differ per sub-block).
        // d=1, nibble=8 (LUT=1), so output[i in sub sb] = (33 + sb) - 32 = 1 + sb.
        let mut bytes = synth_block(1.0, 32, 8);
        // Override scales_l + scales_h such that ls[sb] = 33 + sb (for sb in 0..8 that's 33..41).
        let mut scales_h: u16 = 0;
        let mut scales_l = [0u8; 4];
        for sb in 0..8usize {
            let ls = (33 + sb) as u8;
            let lo4 = ls & 0x0F;
            let hi2 = (ls >> 4) & 0x3;
            scales_h |= (hi2 as u16) << (2 * sb);
            scales_l[sb / 2] |= lo4 << (4 * (sb & 1));
        }
        bytes[2..4].copy_from_slice(&scales_h.to_le_bytes());
        bytes[4..8].copy_from_slice(&scales_l);

        let mut out = vec![0.0_f32; 256];
        dequantize_to_f32(&bytes, &mut out);
        for sb in 0..8 {
            let expected = (1 + sb) as f32;
            let off = sb * 32;
            assert!(out[off..off + 32].iter().all(|v| *v == expected),
                "sb={sb} expected {expected}, got {:?}", &out[off..off + 4]);
        }
    }
}
