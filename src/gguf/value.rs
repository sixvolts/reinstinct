use crate::gguf::error::Result;
use crate::gguf::reader::Reader;
use crate::gguf::types::GgufValueType;

/// A parsed GGUF metadata value.
///
/// Arrays carry their element type alongside the values so consumers can
/// distinguish e.g. `Array(U32, [])` from `Array(String, [])`.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),   I8(i8),
    U16(u16), I16(i16),
    U32(u32), I32(i32),
    U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    String(String),
    Array { element_type: GgufValueType, values: Vec<MetaValue> },
}

impl MetaValue {
    /// Parse a value of the given type from the reader's current position.
    pub fn read(reader: &mut Reader<'_>, ty: GgufValueType) -> Result<Self> {
        Ok(match ty {
            GgufValueType::U8     => Self::U8(reader.read_u8()?),
            GgufValueType::I8     => Self::I8(reader.read_i8()?),
            GgufValueType::U16    => Self::U16(reader.read_u16()?),
            GgufValueType::I16    => Self::I16(reader.read_i16()?),
            GgufValueType::U32    => Self::U32(reader.read_u32()?),
            GgufValueType::I32    => Self::I32(reader.read_i32()?),
            GgufValueType::U64    => Self::U64(reader.read_u64()?),
            GgufValueType::I64    => Self::I64(reader.read_i64()?),
            GgufValueType::F32    => Self::F32(reader.read_f32()?),
            GgufValueType::F64    => Self::F64(reader.read_f64()?),
            GgufValueType::Bool   => Self::Bool(reader.read_bool()?),
            GgufValueType::String => Self::String(reader.read_string()?),
            GgufValueType::Array  => {
                let elem_raw = reader.read_u32()?;
                let element_type = GgufValueType::try_from_u32(elem_raw)?;
                let len = reader.read_u64()? as usize;
                let mut values = Vec::with_capacity(len.min(64 * 1024));
                for _ in 0..len {
                    values.push(Self::read(reader, element_type)?);
                }
                Self::Array { element_type, values }
            }
        })
    }

    /// Coerce to u32 from any integer variant (signed or unsigned) when the
    /// value fits and is non-negative. GGUF metadata mixes signed/unsigned
    /// even for "obviously unsigned" things like RoPE dimension sections,
    /// so accepting both keeps callers from caring.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v)  => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U32(v) => Some(*v),
            Self::U64(v) => u32::try_from(*v).ok(),
            Self::I8(v)  => u32::try_from(*v).ok(),
            Self::I16(v) => u32::try_from(*v).ok(),
            Self::I32(v) => u32::try_from(*v).ok(),
            Self::I64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v)  => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I8(v)  => u64::try_from(*v).ok(),
            Self::I16(v) => u64::try_from(*v).ok(),
            Self::I32(v) => u64::try_from(*v).ok(),
            Self::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<(GgufValueType, &[MetaValue])> {
        match self {
            Self::Array { element_type, values } => Some((*element_type, values.as_slice())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_string_value() {
        let s = "qwen3";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        let mut r = Reader::new(&buf);
        let v = MetaValue::read(&mut r, GgufValueType::String).unwrap();
        assert_eq!(v.as_str(), Some("qwen3"));
    }

    #[test]
    fn parses_typed_array() {
        // Array of three u32: [10, 20, 30]
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes());     // element_type = U32
        buf.extend_from_slice(&3u64.to_le_bytes());     // length
        for x in [10u32, 20, 30] {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        let mut r = Reader::new(&buf);
        let v = MetaValue::read(&mut r, GgufValueType::Array).unwrap();
        match v {
            MetaValue::Array { element_type, values } => {
                assert_eq!(element_type, GgufValueType::U32);
                assert_eq!(values.len(), 3);
                assert_eq!(values[1].as_u32(), Some(20));
            }
            _ => panic!("expected array"),
        }
    }
}
