//! Q3_K: 3.4375 bpw K-quant, super-block of 256.
//!
//! 110-byte block / 256 weights:
//!   u8    hmask[32]    the high (3rd) bit of each quant, one bit per weight
//!   u8    qs[64]       the low 2 bits, four weights per byte
//!   u8    scales[12]   16 six-bit sub-block scales, bit-packed
//!   fp16  d            super-block scale
//!
//! 16 sub-blocks of 16 weights. Per-weight: `w = d * (sc - 32) * (q3 - 4)`
//! with `q3` in 0..7 and `sc` in 0..63.
//!
//! Unlike IQ3_S / IQ4_NL there is no exact Q8_0 transcode: a Q8_0 block is
//! 32 weights under one scale, and a Q3_K block pairs two 16-weight
//! sub-blocks with independent scales. The loader dequantizes and
//! requantizes to Q8_0 instead (`q8_0::quantize_from_f32`). That is not
//! exact, but requantizing a 3-bit source at 8 bits per 32-block loses
//! nothing measurable: the Q8_0 step is <=1/254 of the block's peak, far
//! inside the source's own quantization error.

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 110;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ3_K {
    pub hmask: [u8; 32],
    pub qs: [u8; 64],
    pub scales: [u8; 12],
    pub d: u16,
}

const _: () = assert!(std::mem::size_of::<BlockQ3_K>() == BYTES_PER_BLOCK);

/// Unpack the 16 six-bit scales. A literal port of the `aux[4]` / kmask
/// shuffle in ggml's `dequantize_row_q3_K`, kept verbatim so it cannot
/// drift from the reference.
fn unpack_scales(scales: &[u8; 12]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    aux[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    aux[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut out = [0i8; 16];
    for (i, w) in aux.iter().enumerate() {
        for (j, b) in w.to_le_bytes().iter().enumerate() {
            out[i * 4 + j] = *b as i8;
        }
    }
    out
}

pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);
    let blocks: &[BlockQ3_K] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);

    for (b, ob) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        let d_all = f16_to_f32(b.d);
        let scales = unpack_scales(&b.scales);
        let mut y = 0usize;      // output cursor
        let mut is = 0usize;     // scale cursor
        let mut m: u8 = 1;       // hmask bit for this 2-bit shift level
        for n in (0..BLOCK_SIZE).step_by(128) {
            let q = &b.qs[n / 4..n / 4 + 32];
            let mut shift = 0u32;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let lo = ((q[l] >> shift) & 3) as i32;
                    let hi = if b.hmask[l] & m != 0 { 0 } else { 4 };
                    ob[y] = dl * (lo - hi) as f32;
                    y += 1;
                }
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let lo = ((q[l + 16] >> shift) & 3) as i32;
                    let hi = if b.hmask[l + 16] & m != 0 { 0 } else { 4 };
                    ob[y] = dl * (lo - hi) as f32;
                    y += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    /// A 6-bit scale of 32 is 0b10_0000: low nibble 0, high 2-bit field
    /// 10. ggml packs the low nibbles of scales 0..7 into bytes 0..7 and the
    /// high fields four per byte into bytes 8..11 — so sixteen 32s are
    /// bytes 0..7 = 0x00 and bytes 8..11 = 0xAA. The unpacker must recover
    /// 32 in all sixteen slots, and a block whose scales are all 32 has
    /// factor (32-32) = 0 everywhere, so it dequantizes to zeros whatever
    /// the quants are.
    #[test]
    fn scale_unpack_and_zero_factor_block() {
        let mut all32 = [0u8; 12];
        for i in 8..12 { all32[i] = 0xAA; }
        assert_eq!(unpack_scales(&all32), [32i8; 16]);

        let mut bytes = vec![0u8; BYTES_PER_BLOCK];
        bytes[108..110].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        bytes[96..108].copy_from_slice(&all32);
        for i in 0..96 { bytes[i] = 0xA5; }   // arbitrary quants and hmask
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequantize_to_f32(&bytes, &mut out);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    /// Requantizing to Q8_0 must stay inside the source's own precision.
    #[test]
    fn requantize_to_q8_0_is_within_source_error() {
        const N: usize = 32;
        let mut seed: u64 = 0x0A3E_0000;
        let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                           (seed >> 56) as u8 };
        let mut bytes = vec![0u8; N * BYTES_PER_BLOCK];
        for b in 0..N {
            let off = b * BYTES_PER_BLOCK;
            for i in 0..108 { bytes[off + i] = rng(); }
            bytes[off + 108..off + 110].copy_from_slice(
                &f32_to_f16(0.02 * (1.0 + b as f32 / 32.0)).to_le_bytes());
        }
        let n = N * BLOCK_SIZE;
        let mut direct = vec![0.0f32; n];
        dequantize_to_f32(&bytes, &mut direct);
        let q8 = crate::quant::q8_0::quantize_from_f32(&direct);
        let mut via = vec![0.0f32; n];
        crate::quant::q8_0::dequantize_to_f32(&q8, &mut via);
        // Per 32-block the Q8_0 step is peak/127. Rounding contributes
        // half a step; the scale being stored as fp16 adds up to
        // 127 * 2^-11 ~ 0.06 of a step on the largest |q|. Bound at 0.6.
        for (blk, (a, b)) in direct.chunks(32).zip(via.chunks(32)).enumerate() {
            let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let step = peak / 127.0;
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= step * 0.6 + 1e-7,
                    "block {blk}: |{x} - {y}| exceeds 0.6 of a Q8_0 step ({step})");
            }
        }
    }
}
