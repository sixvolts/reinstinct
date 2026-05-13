use crate::gguf::error::{GgufError, Result};
use crate::gguf::reader::Reader;
use crate::gguf::types::GgmlType;

pub const MAX_DIMS: usize = 4;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: GgmlType,
    n_dims: u8,
    dims: [u64; MAX_DIMS],
    /// Byte offset from the start of the tensor-data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn shape(&self) -> &[u64] {
        &self.dims[..self.n_dims as usize]
    }

    pub fn n_elements(&self) -> u64 {
        self.shape().iter().product()
    }

    /// Size of this tensor's on-disk payload in bytes, computed from shape and type.
    pub fn byte_size(&self) -> Result<u64> {
        let block_elems = self.ggml_type.block_size_elements();
        let n = self.n_elements();
        if n % block_elems != 0 {
            return Err(GgufError::UnalignedTensor {
                name: self.name.clone(),
                ty: self.ggml_type,
                n_elements: n,
                block_size: block_elems,
            });
        }
        Ok((n / block_elems) * self.ggml_type.bytes_per_block())
    }

    pub(crate) fn parse(reader: &mut Reader<'_>) -> Result<Self> {
        let name = reader.read_string()?;
        let n_dims_u32 = reader.read_u32()?;
        if n_dims_u32 == 0 || n_dims_u32 as usize > MAX_DIMS {
            return Err(GgufError::BadShape { name, n_dims: n_dims_u32 });
        }
        let n_dims = n_dims_u32 as u8;
        let mut dims = [1u64; MAX_DIMS];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = reader.read_u64()?;
        }
        let ty_raw = reader.read_u32()?;
        let ggml_type = GgmlType::try_from_u32(ty_raw)?;
        let offset = reader.read_u64()?;
        Ok(Self { name, ggml_type, n_dims, dims, offset })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_size_q4k_matches_design_doc() {
        // A Q4_K tensor of shape [256, 4] = 1024 elements
        // = 1024 / 256 = 4 blocks × 144 bytes/block = 576 bytes.
        let t = TensorInfo {
            name: "test".into(),
            ggml_type: GgmlType::Q4_K,
            n_dims: 2,
            dims: [256, 4, 1, 1],
            offset: 0,
        };
        assert_eq!(t.byte_size().unwrap(), 576);
    }

    #[test]
    fn unaligned_tensor_errors() {
        // 100 elements is not a multiple of Q4_K block size (256).
        let t = TensorInfo {
            name: "bad".into(),
            ggml_type: GgmlType::Q4_K,
            n_dims: 1,
            dims: [100, 1, 1, 1],
            offset: 0,
        };
        assert!(matches!(
            t.byte_size().unwrap_err(),
            GgufError::UnalignedTensor { .. }
        ));
    }
}
