use crate::gguf::error::{GgufError, Result};

/// Bounds-checked little-endian cursor over a byte slice.
///
/// GGUF is little-endian on every platform regardless of host endianness.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.need(n)?;
        self.pos += n;
        Ok(())
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            Err(GgufError::Truncated {
                offset: self.pos,
                needed: n,
                available: self.remaining(),
            })
        } else {
            Ok(())
        }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.need(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(&self.buf[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    pub fn read_u8(&mut self)  -> Result<u8>  { Ok(u8::from_le_bytes(self.read_array::<1>()?)) }
    pub fn read_i8(&mut self)  -> Result<i8>  { Ok(i8::from_le_bytes(self.read_array::<1>()?)) }
    pub fn read_u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.read_array::<2>()?)) }
    pub fn read_i16(&mut self) -> Result<i16> { Ok(i16::from_le_bytes(self.read_array::<2>()?)) }
    pub fn read_u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.read_array::<4>()?)) }
    pub fn read_i32(&mut self) -> Result<i32> { Ok(i32::from_le_bytes(self.read_array::<4>()?)) }
    pub fn read_u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.read_array::<8>()?)) }
    pub fn read_i64(&mut self) -> Result<i64> { Ok(i64::from_le_bytes(self.read_array::<8>()?)) }
    pub fn read_f32(&mut self) -> Result<f32> { Ok(f32::from_le_bytes(self.read_array::<4>()?)) }
    pub fn read_f64(&mut self) -> Result<f64> { Ok(f64::from_le_bytes(self.read_array::<8>()?)) }

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// GGUF string: `u64` byte-length followed by UTF-8 bytes (no NUL).
    pub fn read_string(&mut self) -> Result<String> {
        let offset = self.pos;
        let len = self.read_u64()? as usize;
        self.need(len)?;
        let bytes = &self.buf[self.pos..self.pos + len];
        let s = std::str::from_utf8(bytes)
            .map_err(|source| GgufError::BadUtf8 { offset, source })?
            .to_owned();
        self.pos += len;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_primitives() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        buf.extend_from_slice(&(-1i64).to_le_bytes());
        buf.extend_from_slice(&3.5f32.to_le_bytes());
        buf.push(1);

        let mut r = Reader::new(&buf);
        assert_eq!(r.read_u32().unwrap(), 0xDEADBEEF);
        assert_eq!(r.read_i64().unwrap(), -1);
        assert_eq!(r.read_f32().unwrap(), 3.5);
        assert!(r.read_bool().unwrap());
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn read_string_decodes_utf8() {
        let s = "héllo";
        let bytes = s.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(bytes);

        let mut r = Reader::new(&buf);
        assert_eq!(r.read_string().unwrap(), s);
    }

    #[test]
    fn truncation_is_reported() {
        let buf = [0u8; 3];
        let mut r = Reader::new(&buf);
        assert!(matches!(
            r.read_u32().unwrap_err(),
            GgufError::Truncated { needed: 4, available: 3, .. }
        ));
    }
}
