//! GGUF v3 file parser and tensor loader.
//!
//! The on-disk format is, in order:
//!   1. 4-byte magic "GGUF"
//!   2. u32 version (we accept v3 only)
//!   3. u64 tensor_count
//!   4. u64 metadata_kv_count
//!   5. metadata_kv_count × (key: string, value_type: u32, value: dynamic)
//!   6. tensor_count × tensor_info (name, n_dims, dims[], type, offset)
//!   7. zero-padding up to `general.alignment` (default 32)
//!   8. raw tensor data; each tensor lives at `data_section + tensor.offset`

pub mod error;
pub mod reader;
pub mod tensor_info;
pub mod types;
pub mod value;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

pub use error::{GgufError, Result};
pub use tensor_info::{TensorInfo, MAX_DIMS};
pub use types::{GgmlType, GgufValueType};
pub use value::MetaValue;

use reader::Reader;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian (47 47 55 46)
const SUPPORTED_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct GgufFile {
    mmap: Mmap,
    pub header: Header,
    pub metadata: Vec<(String, MetaValue)>,
    metadata_index: HashMap<String, usize>,
    pub tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    pub alignment: u64,
    /// Absolute byte offset within the mmap where the tensor-data section begins.
    pub data_section_offset: u64,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let mut r = Reader::new(&mmap[..]);

        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic { got: magic });
        }
        let version = r.read_u32()?;
        if version != SUPPORTED_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = r.read_u64()?;
        let metadata_kv_count = r.read_u64()?;
        let header = Header { version, tensor_count, metadata_kv_count };

        let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
        let mut metadata_index = HashMap::with_capacity(metadata_kv_count as usize);
        for i in 0..metadata_kv_count {
            let key = r.read_string()?;
            let vt_raw = r.read_u32()?;
            let vt = GgufValueType::try_from_u32(vt_raw)?;
            let value = MetaValue::read(&mut r, vt)?;
            metadata_index.insert(key.clone(), i as usize);
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut tensor_index = HashMap::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let info = TensorInfo::parse(&mut r)?;
            tensor_index.insert(info.name.clone(), i as usize);
            tensors.push(info);
        }

        let alignment = metadata_index
            .get("general.alignment")
            .and_then(|&i| metadata[i].1.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        let data_section_offset = align_up(r.position() as u64, alignment);

        let file_len = mmap.len() as u64;
        for t in &tensors {
            let size = t.byte_size()?;
            let abs_offset = data_section_offset + t.offset;
            if abs_offset + size > file_len {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: abs_offset,
                    size,
                    file_len,
                });
            }
        }

        Ok(Self {
            mmap,
            header,
            metadata,
            metadata_index,
            tensors,
            tensor_index,
            alignment,
            data_section_offset,
        })
    }

    pub fn metadata_get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata_index.get(key).map(|&i| &self.metadata[i].1)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Zero-copy view of a tensor's on-disk bytes.
    pub fn tensor_data(&self, name: &str) -> Result<Option<&[u8]>> {
        let Some(info) = self.tensor(name) else { return Ok(None); };
        let size = info.byte_size()? as usize;
        let start = (self.data_section_offset + info.offset) as usize;
        Ok(Some(&self.mmap[start..start + size]))
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    if align <= 1 { v } else { v.div_ceil(align) * align }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 1), 100);
    }
}
