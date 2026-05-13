//! IEEE 754 binary16 (FP16) and bfloat16 conversions to/from f32.
//!
//! Avoids pulling in the `half` crate — these are short, well-defined,
//! and we only need conversion (no arithmetic) at this layer. HIP kernels
//! will use native FP16 hardware paths instead.

/// FP16 → f32. Handles subnormals, infinities, and NaN per IEEE 754.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign     = (bits & 0x8000) as u32;
    let exponent = (bits & 0x7C00) as u32;
    let mantissa = (bits & 0x03FF) as u32;

    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            sign << 16
        } else {
            // Subnormal: shift mantissa left until the implicit bit (0x0400)
            // appears, tracking the count. FP16 subnormals have unbiased
            // exponent -14; each shift extracts one factor of 2.
            let mut m = mantissa;
            let mut shifts: i32 = 0;
            while (m & 0x0400) == 0 {
                m <<= 1;
                shifts += 1;
            }
            m &= 0x03FF;
            let biased_f32 = ((-14 - shifts) + 127) as u32;
            (sign << 16) | (biased_f32 << 23) | (m << 13)
        }
    } else if exponent == 0x7C00 {
        // Inf or NaN: keep mantissa pattern, propagate.
        (sign << 16) | 0x7F800000 | (mantissa << 13)
    } else {
        let exp_f32 = (exponent >> 10) + (127 - 15);
        (sign << 16) | (exp_f32 << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

/// BF16 → f32. Trivial: bf16 is the high 16 bits of the f32 representation.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → FP16 (round-to-nearest-even, with proper inf/NaN/subnormal handling).
/// Used in tests; production loaders only need the other direction.
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa_f32 = bits & 0x007F_FFFF;

    if exp_f32 == 0xFF {
        // Inf or NaN.
        let mantissa_f16 = if mantissa_f32 != 0 { 0x0200 } else { 0 };
        return sign | 0x7C00 | mantissa_f16;
    }
    let unbiased = exp_f32 - 127;
    if unbiased >= 16 {
        return sign | 0x7C00; // Overflow → inf.
    }
    if unbiased < -24 {
        return sign; // Underflow → ±0.
    }
    if unbiased < -14 {
        // Subnormal half: shift mantissa right.
        let m = (mantissa_f32 | 0x0080_0000) >> ((-14 - unbiased) + 13) as u32;
        return sign | (m as u16);
    }
    let exp_f16 = ((unbiased + 15) as u16) << 10;
    let mantissa_f16 = (mantissa_f32 >> 13) as u16;
    // Round-to-nearest-even on the truncated bits.
    let round_bit = (mantissa_f32 >> 12) & 1;
    let sticky    = mantissa_f32 & 0x0FFF;
    let mut out   = sign | exp_f16 | mantissa_f16;
    if round_bit != 0 && (sticky != 0 || (mantissa_f16 & 1) != 0) {
        out = out.wrapping_add(1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_roundtrip_known_values() {
        // (fp16 bits, expected f32 value)
        let cases: &[(u16, f32)] = &[
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3C00, 1.0),
            (0xBC00, -1.0),
            (0x4000, 2.0),
            (0x4200, 3.0),
            (0x3800, 0.5),
            (0x7BFF, 65504.0),     // FP16 max
            (0xFBFF, -65504.0),
            (0x7C00, f32::INFINITY),
            (0xFC00, f32::NEG_INFINITY),
            (0x0001, 5.960_464_5e-8), // smallest positive subnormal
        ];
        for &(bits, expected) in cases {
            let got = f16_to_f32(bits);
            assert_eq!(got.to_bits(), expected.to_bits(),
                "fp16 0x{bits:04X} → got {got}, expected {expected}");
        }
    }

    #[test]
    fn fp16_nan_propagates() {
        let nan = f16_to_f32(0x7E00);
        assert!(nan.is_nan());
    }

    #[test]
    fn bf16_roundtrip_known_values() {
        // BF16 is the high 16 bits of f32; round-trip is exact for values
        // representable in BF16 and lossy elsewhere — but always trivial bits.
        for f in [1.0_f32, -1.0, 0.0, 256.0, -0.5] {
            let bf = (f.to_bits() >> 16) as u16;
            assert_eq!(bf16_to_f32(bf), f);
        }
    }

    #[test]
    fn fp16_round_trip_through_f32_to_f16() {
        for bits in [0x0000_u16, 0x3C00, 0xBC00, 0x4200, 0x3800, 0x7BFF, 0xFBFF] {
            let f = f16_to_f32(bits);
            assert_eq!(f32_to_f16(f), bits, "round trip failed at 0x{bits:04X}");
        }
    }
}
