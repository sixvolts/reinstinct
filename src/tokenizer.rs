//! Decode-only GPT2-style BPE tokenizer for Qwen 3.5 (and similar GGUF
//! models with `tokenizer.ggml.model == "gpt2"`).
//!
//! This is enough to turn `generate_text`'s output token ids into the
//! UTF-8 string the model intended. Encoding (text → tokens) is a
//! separate, much heavier piece of work (pre-tokenizer regex + BPE
//! merge algorithm) and is not implemented here.
//!
//! GPT2 byte-level BPE: each input byte 0..255 is first mapped to a
//! "printable" Unicode codepoint via the `bytes_to_unicode` table, so
//! the raw vocab strings are guaranteed to be valid UTF-8. To decode,
//! we walk the concatenated vocab string and invert that mapping.

use std::collections::HashMap;

use crate::gguf::{GgufFile, MetaValue};

/// Build the GPT2 byte→unicode permutation. Identical to the upstream
/// `bytes_to_unicode()` Python helper. Returns a 256-entry array
/// indexed by byte value.
fn bytes_to_unicode() -> [char; 256] {
    // Start with the printable-ASCII + Latin-1 supplement subset that
    // doesn't need remapping.
    let mut bs: Vec<u32> = (b'!'..=b'~').map(|b| b as u32).collect();
    bs.extend((0xA1u32..=0xACu32).chain(0xAEu32..=0xFFu32));

    let mut cs: Vec<u32> = bs.clone();
    let mut n: u32 = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = [' '; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

pub struct Tokenizer {
    /// Vocab table: token_id → vocab string (in GPT2 byte-encoded form).
    tokens: Vec<String>,
    /// Reverse map of `bytes_to_unicode`: codepoint → original byte.
    byte_decoder: HashMap<char, u8>,
    pub eos_id: u32,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let model = gguf.metadata_get("tokenizer.ggml.model")
            .ok_or("missing tokenizer.ggml.model")?;
        match model {
            MetaValue::String(s) if s == "gpt2" => {}
            other => return Err(format!("only gpt2-style tokenizers supported, got {other:?}")),
        }

        let tokens_meta = gguf.metadata_get("tokenizer.ggml.tokens")
            .ok_or("missing tokenizer.ggml.tokens")?;
        let tokens: Vec<String> = match tokens_meta {
            MetaValue::Array { values, .. } => values.iter().map(|v| match v {
                MetaValue::String(s) => Ok(s.clone()),
                other => Err(format!("non-string token: {other:?}")),
            }).collect::<Result<Vec<_>, _>>()?,
            other => return Err(format!("tokens not an array: {other:?}")),
        };

        let eos_id = gguf.metadata_get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .ok_or("missing tokenizer.ggml.eos_token_id")?;

        // Build the byte decoder: every char that bytes_to_unicode emits
        // maps back to its source byte; chars that aren't in the table
        // are left alone (they're literal UTF-8 from the vocab).
        let b2u = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, c) in b2u.iter().enumerate() {
            byte_decoder.insert(*c, b as u8);
        }

        Ok(Self { tokens, byte_decoder, eos_id })
    }

    pub fn vocab_size(&self) -> usize { self.tokens.len() }

    /// Look up the raw vocab string (still in GPT2-encoded form) for a
    /// single token id. Returns `<unk:N>` for out-of-range ids.
    pub fn token_str(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(|s| s.as_str()).unwrap_or("<unk>")
    }

    /// Decode a sequence of token ids to a UTF-8 string. Token strings
    /// are concatenated and then the GPT2 byte permutation is inverted
    /// to recover the original byte stream, which is finally lossy-UTF-8-
    /// decoded back to a Rust String.
    pub fn decode(&self, ids: &[u32]) -> String {
        // First concatenate the vocab pieces.
        let mut joined = String::new();
        for &id in ids {
            joined.push_str(self.token_str(id));
        }

        // Walk chars; chars in the byte_decoder map back to their byte.
        // Anything else (e.g. pieces that contain literal UTF-8 from
        // special added tokens like <|endoftext|>) we keep verbatim.
        let mut bytes: Vec<u8> = Vec::with_capacity(joined.len());
        for c in joined.chars() {
            if let Some(&b) = self.byte_decoder.get(&c) {
                bytes.push(b);
            } else {
                // Likely a special token (e.g. <|...|>) that wasn't byte-
                // encoded in the first place. Push its UTF-8 bytes.
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var_os("HOME")?;
        let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
        p.exists().then_some(p)
    }

    #[test]
    fn loads_qwen35_tokenizer() {
        let Some(p) = fixture_path() else { eprintln!("skip"); return };
        let g = GgufFile::open(&p).unwrap();
        let tok = Tokenizer::from_gguf(&g).expect("load tokenizer");
        eprintln!("vocab = {}, eos = {}", tok.vocab_size(), tok.eos_id);
        assert!(tok.vocab_size() > 200_000);
    }

    #[test]
    fn decodes_known_token_ids() {
        let Some(p) = fixture_path() else { eprintln!("skip"); return };
        let g = GgufFile::open(&p).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        // Token 198 in GPT2-style vocab is typically a newline ("Ċ" in
        // byte-encoded form → 0x0A). Check that it round-trips.
        let s = tok.decode(&[198]);
        assert_eq!(s, "\n", "token 198 should decode to newline, got {s:?}");
        // Token 220 should be a leading space ("Ġ" → 0x20).
        let s = tok.decode(&[220]);
        assert_eq!(s, " ", "token 220 should decode to space, got {s:?}");
    }

    #[test]
    fn round_trip_simple_sequence() {
        // Just verify decoding a known sequence produces non-empty,
        // non-control-character text. Real BPE round-trips need the
        // encoder, which we don't have.
        let Some(p) = fixture_path() else { eprintln!("skip"); return };
        let g = GgufFile::open(&p).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        // 198=newline, 220=space, 16=number "1", 17=number "2".
        // Let's just check decoding doesn't panic and produces something.
        let s = tok.decode(&[198, 220, 16, 17]);
        eprintln!("decode([198,220,16,17]) = {s:?}");
        assert!(!s.is_empty());
    }
}
