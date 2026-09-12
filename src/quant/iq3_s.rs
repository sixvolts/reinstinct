//! IQ3_S: 3.4375 bpw importance-quantized 3-bit, super-block of 256.
//!
//! 110-byte block / 256 weights:
//!   fp16  d              super-block scale
//!   u8    qs[64]         low 8 bits of a 9-bit grid index, one per 4 weights
//!   u8    qh[8]          the 9th index bit, one bit per 4 weights
//!   u8    signs[32]      one sign bit per weight
//!   u8    scales[4]      two 4-bit sub-scales per byte, one per 32 weights
//!
//! Each 4-weight group is a codebook entry: `iq3s_grid[idx]` unpacks to 4
//! bytes in 1..15, and the sign bits flip them. Per 32-weight sub-block
//! `db = d * (1 + 2 * scale4)`, so `w = db * ±grid_byte`.
//!
//! That last line is why this transcodes to Q8_0 exactly: a sub-block is
//! 32 weights sharing one scale, each a signed value in -15..15 — which is
//! a Q8_0 block term for term, with `d = db` and `q = ±grid_byte`. Only
//! `db` moves, by one fp16 rounding. See [`transcode_to_q8_0`].

use bytemuck::{Pod, Zeroable};

use crate::quant::half::f16_to_f32;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 110;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIQ3_S {
    pub d: u16,
    pub qs: [u8; 64],
    pub qh: [u8; 8],
    pub signs: [u8; 32],
    pub scales: [u8; 4],
}

const _: () = assert!(std::mem::size_of::<BlockIQ3_S>() == BYTES_PER_BLOCK);

/// ggml's `iq3s_grid`: 512 entries, each packing 4 bytes (little-endian)
/// in 1..15. Index = `qs` byte | (a `qh` bit << 8).
pub const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// `kmask_iq2xs` — which sign bit belongs to which of the 8 weights a
/// sign byte covers.
const KMASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Walk a block yielding `(sub_block_scale_db, [i8; 32])` for each of the
/// eight 32-weight sub-blocks. Shared by the dequant and the transcode so
/// they cannot disagree.
fn sub_blocks(b: &BlockIQ3_S) -> [(f32, [i8; 32]); 8] {
    let d = f16_to_f32(b.d);
    let mut out = [(0.0f32, [0i8; 32]); 8];
    let mut qs = 0usize;      // byte cursor into b.qs
    let mut sg = 0usize;      // byte cursor into b.signs
    let mut qh = 0usize;      // byte cursor into b.qh
    // Two 32-weight sub-blocks per outer step; they share a scales byte.
    for pair in 0..4usize {
        let db1 = d * (1.0 + 2.0 * (b.scales[pair] & 0x0F) as f32);
        let db2 = d * (1.0 + 2.0 * (b.scales[pair] >> 4) as f32);
        for (half, db) in [(0usize, db1), (1usize, db2)] {
            let mut vals = [0i8; 32];
            for l in 0..4usize {
                let idx1 = b.qs[qs + 2 * l]     as usize | (((b.qh[qh] as usize) << (8 - 2 * l)) & 256);
                let idx2 = b.qs[qs + 2 * l + 1] as usize | (((b.qh[qh] as usize) << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[idx1].to_le_bytes();
                let g2 = IQ3S_GRID[idx2].to_le_bytes();
                let s = b.signs[sg + l];
                for j in 0..4usize {
                    let v1 = g1[j] as i8;
                    let v2 = g2[j] as i8;
                    vals[l * 8 + j]     = if s & KMASK[j]     != 0 { -v1 } else { v1 };
                    vals[l * 8 + j + 4] = if s & KMASK[j + 4] != 0 { -v2 } else { v2 };
                }
            }
            out[pair * 2 + half] = (db, vals);
            qs += 8;
            sg += 4;
            qh += 1;
        }
    }
    out
}

pub fn dequantize_to_f32(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % BLOCK_SIZE, 0);
    let n_blocks = out.len() / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);
    let blocks: &[BlockIQ3_S] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);
    for (b, ob) in blocks.iter().zip(out.chunks_exact_mut(BLOCK_SIZE)) {
        for (s, (db, vals)) in sub_blocks(b).iter().enumerate() {
            for i in 0..32 { ob[s * 32 + i] = db * vals[i] as f32; }
        }
    }
}

/// Transcode IQ3_S bytes into the equivalent Q8_0 bytes — a relabelling,
/// not a requantization. Each 32-weight sub-block already IS a Q8_0
/// block: one scale, 32 signed small ints. Only the scale is rounded, to
/// fp16, ~2⁻¹¹ relative. Costs 3.44 → 8.5 bpw on the tensors involved,
/// and buys the full repacked-int8 kernel set (matvec, batched, MMQ).
pub fn transcode_to_q8_0(bytes: &[u8], n_weights: usize) -> Vec<u8> {
    use crate::quant::half::f32_to_f16;
    use crate::quant::q8_0::{BlockQ8_0, BYTES_PER_BLOCK as Q8_BYTES};
    assert_eq!(n_weights % BLOCK_SIZE, 0);
    let n_blocks = n_weights / BLOCK_SIZE;
    assert!(bytes.len() >= n_blocks * BYTES_PER_BLOCK);
    let blocks: &[BlockIQ3_S] =
        bytemuck::cast_slice(&bytes[..n_blocks * BYTES_PER_BLOCK]);
    let mut out = vec![0u8; n_weights / 32 * Q8_BYTES];
    let dst: &mut [BlockQ8_0] = bytemuck::cast_slice_mut(&mut out);
    for (bi, b) in blocks.iter().enumerate() {
        for (s, (db, vals)) in sub_blocks(b).iter().enumerate() {
            let o = &mut dst[bi * 8 + s];
            o.d = f32_to_f16(*db);   // db >= 0 by construction: d*(1+2*s4) with d>=0
            o.qs = *vals;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    #[test]
    fn transcode_matches_direct_dequant() {
        const N: usize = 48;
        let mut seed: u64 = 0x1A35_0000;
        let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                           (seed >> 56) as u8 };
        let mut bytes = vec![0u8; N * BYTES_PER_BLOCK];
        for b in 0..N {
            let off = b * BYTES_PER_BLOCK;
            let d = 10f32.powi(b as i32 % 4 - 2) * (1.0 + b as f32 / 48.0);
            bytes[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            for i in 2..BYTES_PER_BLOCK { bytes[off + i] = rng(); }
        }
        let n = N * BLOCK_SIZE;
        let mut direct = vec![0.0f32; n];
        dequantize_to_f32(&bytes, &mut direct);
        let q8 = transcode_to_q8_0(&bytes, n);
        let mut via = vec![0.0f32; n];
        crate::quant::q8_0::dequantize_to_f32(&q8, &mut via);
        let mut max_rel = 0.0f32;
        for (a, b) in direct.iter().zip(&via) {
            max_rel = max_rel.max((a - b).abs() / a.abs().max(1e-30));
        }
        assert!(max_rel < 5e-4, "iq3_s transcode max_rel {max_rel:.3e}");
    }
}
