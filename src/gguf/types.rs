use crate::gguf::error::{GgufError, Result};

/// Recognized ggml tensor types.
///
/// Discriminants match the upstream `ggml_type` enum so they can be cast
/// directly to/from the `uint32` stored in GGUF tensor-info entries.
///
/// Recognition (`try_from_u32`) is broader than execution (`has_dequant_kernel`).
/// In practice Unsloth UD-XL files mix in IQ-formats (IQ4_XS, IQ4_NL) for
/// sensitive tensors alongside the K-quants the engine actually runs;
/// recognizing them lets us inspect and validate files even before
/// the matching kernels exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)] // ggml/llama.cpp canonical names (Q4_K etc.)
pub enum GgmlType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ4_NL  = 20,
    IQ4_XS  = 23,
    BF16    = 30,
}

impl GgmlType {
    pub fn try_from_u32(v: u32) -> Result<Self> {
        match v {
            0  => Ok(Self::F32),
            1  => Ok(Self::F16),
            2  => Ok(Self::Q4_0),
            3  => Ok(Self::Q4_1),
            6  => Ok(Self::Q5_0),
            7  => Ok(Self::Q5_1),
            8  => Ok(Self::Q8_0),
            9  => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            20 => Ok(Self::IQ4_NL),
            23 => Ok(Self::IQ4_XS),
            30 => Ok(Self::BF16),
            _  => Err(GgufError::UnknownGgmlType(v)),
        }
    }

    /// Number of weights packed into one on-disk block.
    pub const fn block_size_elements(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
                | Self::Q8_0 | Self::Q8_1 | Self::IQ4_NL => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K
                | Self::Q6_K | Self::Q8_K | Self::IQ4_XS => 256,
        }
    }

    /// On-disk size of one block in bytes (matches ggml `type_traits` table).
    pub const fn bytes_per_block(self) -> u64 {
        match self {
            Self::F32     => 4,
            Self::F16     => 2,
            Self::BF16    => 2,
            Self::Q4_0    => 18,   // fp16 d + 16 nibbles
            Self::Q4_1    => 20,   // fp16 d + fp16 m + 16 nibbles
            Self::Q5_0    => 22,   // fp16 d + 4-byte high bits + 16 nibbles
            Self::Q5_1    => 24,   // fp16 d + fp16 m + 4-byte high bits + 16 nibbles
            Self::Q8_0    => 34,   // fp16 scale + 32 int8
            Self::Q8_1    => 36,   // fp16 d + fp16 s + 32 int8
            Self::Q2_K    => 84,   // 16 scales + 64 quants + fp16 d + fp16 dmin
            Self::Q3_K    => 110,  // 32 hbits + 64 lqs + 12 scales + fp16 d
            Self::Q4_K    => 144,  // 2 fp16 + 12 packed scales + 128 nibbles
            Self::Q5_K    => 176,  // Q4_K + 32 high-bit bytes
            Self::Q6_K    => 210,  // 128 ql + 64 qh + 16 int8 scales + fp16
            Self::Q8_K    => 292,  // fp32 d + 256 int8 + 16 int16 bsums
            Self::IQ4_NL  => 18,   // fp16 d + 16 nibbles indexing 16-entry LUT
            Self::IQ4_XS  => 136,  // fp16 d + u16 scales_h + 4 scales_l + 128 nibbles
        }
    }

    /// True if the engine has a fused dequant-GEMV kernel for this type.
    /// F32/F16/BF16 are weights too, but consumed via type-conversion paths.
    /// IQ-types parse but currently have no kernel — see [[project_overview]].
    pub const fn has_dequant_kernel(self) -> bool {
        matches!(self, Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_0)
    }
}

/// GGUF metadata value-type discriminator (`gguf_metadata_value_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3,
    U32 = 4, I32 = 5, F32 = 6, Bool = 7,
    String = 8, Array = 9,
    U64 = 10, I64 = 11, F64 = 12,
}

impl GgufValueType {
    pub fn try_from_u32(v: u32) -> Result<Self> {
        match v {
            0  => Ok(Self::U8),
            1  => Ok(Self::I8),
            2  => Ok(Self::U16),
            3  => Ok(Self::I16),
            4  => Ok(Self::U32),
            5  => Ok(Self::I32),
            6  => Ok(Self::F32),
            7  => Ok(Self::Bool),
            8  => Ok(Self::String),
            9  => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _  => Err(GgufError::UnknownValueType(v)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_design_doc() {
        // Sanity-check the table in §3 of the design doc.
        assert_eq!(GgmlType::Q4_K.bytes_per_block(), 144);
        assert_eq!(GgmlType::Q5_K.bytes_per_block(), 176);
        assert_eq!(GgmlType::Q6_K.bytes_per_block(), 210);
        assert_eq!(GgmlType::Q8_0.bytes_per_block(), 34);
        assert_eq!(GgmlType::Q8_K.bytes_per_block(), 292);

        // Block element count.
        assert_eq!(GgmlType::Q8_0.block_size_elements(), 32);
        assert_eq!(GgmlType::Q4_K.block_size_elements(), 256);
    }

    #[test]
    fn unknown_ggml_type_errors() {
        assert!(matches!(
            GgmlType::try_from_u32(99).unwrap_err(),
            GgufError::UnknownGgmlType(99)
        ));
    }
}
