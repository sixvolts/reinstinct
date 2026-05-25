//! Block-quantization formats and CPU dequantization oracles.
//!
//! These reference implementations are the correctness oracle for the
//! HIP kernels in `kernels/dequant_*.hip`.

pub mod half;
pub mod iq4_xs;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod turbo3;

use crate::gguf::error::{GgufError, Result};
use crate::gguf::tensor_info::TensorInfo;
use crate::gguf::types::GgmlType;

/// Dispatch dequantization for a tensor's on-disk bytes into a freshly
/// allocated f32 vector. The CPU oracle path used to validate HIP kernels
/// and (later) build a CPU forward-pass reference.
pub fn dequantize_tensor(info: &TensorInfo, tensor_bytes: &[u8]) -> Result<Vec<f32>> {
    let n = info.n_elements() as usize;
    let mut out = vec![0.0_f32; n];
    dequantize_to_f32(info.ggml_type, tensor_bytes, &mut out, info)?;
    Ok(out)
}

/// Dispatch into the right per-type kernel.
///
/// `info` is passed for richer error context (tensor name on failure).
pub fn dequantize_to_f32(
    ty: GgmlType,
    bytes: &[u8],
    out: &mut [f32],
    info: &TensorInfo,
) -> Result<()> {
    match ty {
        GgmlType::F32 => {
            let needed = out.len() * 4;
            if bytes.len() < needed {
                return Err(GgufError::Truncated {
                    offset: 0, needed, available: bytes.len(),
                });
            }
            for (i, chunk) in bytes[..needed].chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            let needed = out.len() * 2;
            if bytes.len() < needed {
                return Err(GgufError::Truncated {
                    offset: 0, needed, available: bytes.len(),
                });
            }
            for (i, chunk) in bytes[..needed].chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                out[i] = half::f16_to_f32(bits);
            }
        }
        GgmlType::BF16 => {
            let needed = out.len() * 2;
            if bytes.len() < needed {
                return Err(GgufError::Truncated {
                    offset: 0, needed, available: bytes.len(),
                });
            }
            for (i, chunk) in bytes[..needed].chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                out[i] = half::bf16_to_f32(bits);
            }
        }
        GgmlType::Q4_K   => q4_k::dequantize_to_f32(bytes, out),
        GgmlType::Q5_K   => q5_k::dequantize_to_f32(bytes, out),
        GgmlType::Q6_K   => q6_k::dequantize_to_f32(bytes, out),
        GgmlType::Q8_0   => q8_0::dequantize_to_f32(bytes, out),
        GgmlType::IQ4_XS => iq4_xs::dequantize_to_f32(bytes, out),
        // Recognized but no oracle: surface the tensor name so the caller knows what's missing.
        ty => return Err(GgufError::UnsupportedGgmlTypeFor { name: info.name.clone(), ty }),
    }
    Ok(())
}
