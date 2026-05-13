use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a GGUF file (bad magic: got {got:#010x}, expected 0x46554747 / \"GGUF\")")]
    BadMagic { got: u32 },

    #[error("unsupported GGUF version: {0} (only v3 is supported)")]
    UnsupportedVersion(u32),

    #[error("truncated read: needed {needed} bytes at offset {offset}, only {available} available")]
    Truncated { offset: usize, needed: usize, available: usize },

    #[error("unknown ggml tensor type: {0}")]
    UnknownGgmlType(u32),

    #[error("unsupported ggml tensor type for this engine: {0:?} (supported: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, Q8_K)")]
    UnsupportedGgmlType(crate::gguf::types::GgmlType),

    #[error("unknown GGUF metadata value type: {0}")]
    UnknownValueType(u32),

    #[error("invalid UTF-8 in GGUF string at offset {offset}: {source}")]
    BadUtf8 { offset: usize, #[source] source: std::str::Utf8Error },

    #[error("tensor `{name}` has invalid shape (n_dims={n_dims}, max 4 supported)")]
    BadShape { name: String, n_dims: u32 },

    #[error("tensor `{name}` element count {n_elements} is not a multiple of block size {block_size} for type {ty:?}")]
    UnalignedTensor {
        name: String,
        ty: crate::gguf::types::GgmlType,
        n_elements: u64,
        block_size: u64,
    },

    #[error("tensor `{name}` data extends past file end (offset={offset}, size={size}, file_len={file_len})")]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        size: u64,
        file_len: u64,
    },
}

pub type Result<T> = std::result::Result<T, GgufError>;
