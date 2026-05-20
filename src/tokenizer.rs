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

/// Qwen2-family pre-tokenizer regex. Qwen 2 / 2.5 / 3 (and the "qwen35"
/// pre type) all use this split pattern; it isolates contractions,
/// letter runs, number runs, punctuation runs, and whitespace.
const QWEN_PRETOKENIZER_REGEX: &str =
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

pub struct Tokenizer {
    /// Vocab table: token_id → vocab string (in GPT2 byte-encoded form).
    tokens: Vec<String>,
    /// Reverse map of `bytes_to_unicode`: codepoint → original byte.
    byte_decoder: HashMap<char, u8>,
    /// Forward byte→unicode permutation (for encoding).
    byte_encoder: [char; 256],
    /// vocab string → token id.
    vocab_map: HashMap<String, u32>,
    /// BPE merge ranks: (left, right) → rank (lower = merged first).
    merge_ranks: HashMap<(String, String), u32>,
    /// Compiled pre-tokenizer regex.
    pre_regex: fancy_regex::Regex,
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
        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(*c, b as u8);
        }

        // Vocab string → id for encode-side lookup.
        let mut vocab_map = HashMap::with_capacity(tokens.len());
        for (id, s) in tokens.iter().enumerate() {
            vocab_map.insert(s.clone(), id as u32);
        }

        // BPE merge ranks from tokenizer.ggml.merges — each entry is
        // "<left> <right>"; the array index is the rank.
        let merge_ranks = match gguf.metadata_get("tokenizer.ggml.merges") {
            Some(MetaValue::Array { values, .. }) => {
                let mut m = HashMap::with_capacity(values.len());
                for (rank, v) in values.iter().enumerate() {
                    if let MetaValue::String(s) = v {
                        // Split on the FIRST space — merge halves can't
                        // themselves contain the byte-encoded space char
                        // (that's "Ġ", not 0x20).
                        if let Some(sp) = s.find(' ') {
                            let l = s[..sp].to_string();
                            let r = s[sp + 1..].to_string();
                            m.insert((l, r), rank as u32);
                        }
                    }
                }
                m
            }
            _ => return Err("missing tokenizer.ggml.merges".into()),
        };

        let pre_regex = fancy_regex::Regex::new(QWEN_PRETOKENIZER_REGEX)
            .map_err(|e| format!("compile pre-tokenizer regex: {e}"))?;

        Ok(Self { tokens, byte_decoder, byte_encoder, vocab_map, merge_ranks,
                  pre_regex, eos_id })
    }

    /// Encode text to token ids. GPT2 byte-level BPE:
    ///   1. pre-tokenize into chunks via the Qwen regex
    ///   2. byte-encode each chunk (UTF-8 bytes → byte_encoder chars)
    ///   3. greedily apply the lowest-rank adjacent merge until none apply
    ///   4. map each surviving piece to its vocab id
    ///
    /// Unknown pieces (shouldn't happen for byte-level BPE — every single
    /// byte char is in the vocab) are skipped with a logged warning.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        // `find_iter` yields the pre-token chunks in order.
        let mut search_from = 0usize;
        while search_from < text.len() {
            let m = match self.pre_regex.find_from_pos(text, search_from) {
                Ok(Some(m)) => m,
                _ => break,
            };
            let chunk = &text[m.start()..m.end()];
            search_from = m.end().max(search_from + 1);
            self.encode_chunk(chunk, &mut out);
        }
        out
    }

    /// BPE-encode one pre-token chunk, appending ids to `out`.
    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) {
        if chunk.is_empty() { return; }

        // Byte-encode: each UTF-8 byte → one byte_encoder char → a
        // one-char String. `word` is the working list of pieces.
        let mut word: Vec<String> = chunk.bytes()
            .map(|b| self.byte_encoder[b as usize].to_string())
            .collect();
        if word.is_empty() { return; }

        // Greedy lowest-rank merge.
        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..word.len().saturating_sub(1) {
                if let Some(&r) = self.merge_ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if r < best_rank { best_rank = r; best_idx = i; }
                }
            }
            if best_idx == usize::MAX { break; }
            // Merge word[best_idx] + word[best_idx+1].
            let merged = format!("{}{}", word[best_idx], word[best_idx + 1]);
            word[best_idx] = merged;
            word.remove(best_idx + 1);
        }

        for piece in &word {
            match self.vocab_map.get(piece) {
                Some(&id) => out.push(id),
                None => eprintln!("tokenizer: piece {piece:?} not in vocab — skipped"),
            }
        }
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

/// SentencePiece-style BPE tokenizer for Gemma 4 (`tokenizer.ggml.model
/// == "gemma4"`). Differs from the GPT-2 `Tokenizer`: spaces are the
/// metaspace char `▁` (U+2581), a dummy `▁` is prepended, and any
/// character absent from the vocab falls back to `<0xXX>` byte tokens.
/// Encoding is merge-rank BPE over the metaspace-transformed text.
pub struct GemmaTokenizer {
    tokens: Vec<String>,
    vocab_map: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    /// byte value → `<0xXX>` token id (for byte fallback on encode).
    byte_to_id: [Option<u32>; 256],
    /// `<0xXX>` token id → byte value (for decode).
    id_to_byte: HashMap<u32, u8>,
    pub bos_id: u32,
    pub eos_id: u32,
}

const METASPACE: char = '\u{2581}';

impl GemmaTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        match gguf.metadata_get("tokenizer.ggml.model") {
            Some(MetaValue::String(s)) if s == "gemma4" => {}
            other => return Err(format!(
                "GemmaTokenizer: expected tokenizer.ggml.model \"gemma4\", got {other:?}")),
        }

        let tokens: Vec<String> = match gguf.metadata_get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array { values, .. }) => values.iter().map(|v| match v {
                MetaValue::String(s) => Ok(s.clone()),
                other => Err(format!("non-string token: {other:?}")),
            }).collect::<Result<Vec<_>, _>>()?,
            other => return Err(format!("tokens not an array: {other:?}")),
        };

        let bos_id = gguf.metadata_get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32()).ok_or("missing tokenizer.ggml.bos_token_id")?;
        let eos_id = gguf.metadata_get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32()).ok_or("missing tokenizer.ggml.eos_token_id")?;

        let mut vocab_map = HashMap::with_capacity(tokens.len());
        for (id, s) in tokens.iter().enumerate() {
            vocab_map.insert(s.clone(), id as u32);
        }

        // Merge ranks — "<left> <right>", array index is the rank. The
        // separator is the only literal space (token strings spell space
        // as the metaspace char), so split on the first space.
        let merge_ranks = match gguf.metadata_get("tokenizer.ggml.merges") {
            Some(MetaValue::Array { values, .. }) => {
                let mut m = HashMap::with_capacity(values.len());
                for (rank, v) in values.iter().enumerate() {
                    if let MetaValue::String(s) = v {
                        if let Some(sp) = s.find(' ') {
                            m.insert((s[..sp].to_string(), s[sp + 1..].to_string()),
                                     rank as u32);
                        }
                    }
                }
                m
            }
            _ => return Err("missing tokenizer.ggml.merges".into()),
        };

        // Byte-fallback tokens are spelled `<0xXX>` (uppercase hex).
        let mut byte_to_id = [None; 256];
        let mut id_to_byte = HashMap::with_capacity(256);
        for b in 0..256usize {
            if let Some(&id) = vocab_map.get(&format!("<0x{b:02X}>")) {
                byte_to_id[b] = Some(id);
                id_to_byte.insert(id, b as u8);
            }
        }

        Ok(Self { tokens, vocab_map, merge_ranks, byte_to_id, id_to_byte, bos_id, eos_id })
    }

    /// Encode text to token ids (no BOS — the caller prepends `bos_id`).
    /// SPM metaspace: prepend a dummy space, replace spaces with `▁`,
    /// split to chars (byte-fallback for non-vocab chars), merge-rank BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let prepared: String = format!(" {text}")
            .chars().map(|c| if c == ' ' { METASPACE } else { c }).collect();

        let mut word: Vec<String> = Vec::new();
        for ch in prepared.chars() {
            let s = ch.to_string();
            if self.vocab_map.contains_key(&s) {
                word.push(s);
            } else {
                let mut buf = [0u8; 4];
                for &b in ch.encode_utf8(&mut buf).as_bytes() {
                    word.push(format!("<0x{b:02X}>"));
                }
            }
        }

        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..word.len().saturating_sub(1) {
                if let Some(&r) = self.merge_ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if r < best_rank { best_rank = r; best_idx = i; }
                }
            }
            if best_idx == usize::MAX { break; }
            let merged = format!("{}{}", word[best_idx], word[best_idx + 1]);
            word[best_idx] = merged;
            word.remove(best_idx + 1);
        }

        let mut out = Vec::with_capacity(word.len());
        for piece in &word {
            match self.vocab_map.get(piece) {
                Some(&id) => out.push(id),
                None => {
                    // Should not happen — fall back to raw bytes.
                    for &b in piece.as_bytes() {
                        if let Some(id) = self.byte_to_id[b as usize] { out.push(id); }
                    }
                }
            }
        }
        out
    }

    pub fn vocab_size(&self) -> usize { self.tokens.len() }

    pub fn token_str(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(|s| s.as_str()).unwrap_or("<unk>")
    }

    /// Look up a literal vocab entry by string — `None` if the token isn't
    /// a single-piece vocab entry. Used by the chat template renderer to
    /// resolve role names (`system`, `user`, `model`) to their atomic ids
    /// without going through SPM's leading-metaspace encoding.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab_map.get(s).copied()
    }

    /// Decode ids to text: byte tokens become raw bytes, the metaspace
    /// char becomes a space, everything else is literal.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(&b) = self.id_to_byte.get(&id) {
                bytes.push(b);
                continue;
            }
            let Some(s) = self.tokens.get(id as usize) else { continue };
            for ch in s.chars() {
                if ch == METASPACE {
                    bytes.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
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

    fn gemma_fixture() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("REINSTINCT_GEMMA_FIXTURE") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var_os("HOME")?;
        let p = PathBuf::from(home)
            .join("models/gemma4-26B/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf");
        p.exists().then_some(p)
    }

    #[test]
    fn gemma_tokenizer_loads_and_round_trips() {
        let Some(p) = gemma_fixture() else { eprintln!("skip: no gemma fixture"); return };
        let g = GgufFile::open(&p).unwrap();
        let tok = GemmaTokenizer::from_gguf(&g).expect("load gemma tokenizer");
        eprintln!("gemma vocab = {}, bos = {}, eos = {}",
                  tok.vocab_size(), tok.bos_id, tok.eos_id);
        for id in [0u32, 1, 2, 3, 105, 106] {
            eprintln!("  token {id:>4} = {:?}", tok.token_str(id));
        }
        // A few byte-fallback tokens.
        for b in [0x41u8, 0x0A, 0xFF] {
            eprintln!("  byte 0x{b:02X} -> id {:?}", tok.byte_to_id[b as usize]);
        }
        // Round-trip: SPM is lossless, decode(encode(x)) == " " + x
        // (the leading space is the dummy metaspace prefix).
        for case in ["Hello, world!", "The quick brown fox.", "numbers 123 + 456",
                     "unicode: café 日本語 🦀"] {
            let ids = tok.encode(case);
            let back = tok.decode(&ids);
            eprintln!("encode({case:?}) = {} ids: {ids:?}\n  decode -> {back:?}",
                      ids.len());
            assert_eq!(back, format!(" {case}"),
                       "round-trip failed for {case:?}");
        }
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
    fn encode_decode_round_trips() {
        let Some(p) = fixture_path() else { eprintln!("skip"); return };
        let g = GgufFile::open(&p).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        // Byte-level BPE is lossless: decode(encode(x)) == x exactly.
        let cases = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "  leading and  multiple   spaces ",
            "numbers 12345 and symbols @#$%^&*()",
            "newlines\nand\ttabs",
            "unicode: café, naïve, 日本語, emoji 🦀",
            "",
        ];
        for case in &cases {
            let ids = tok.encode(case);
            let back = tok.decode(&ids);
            assert_eq!(&back, case,
                "round-trip failed:\n  in:  {case:?}\n  ids: {ids:?}\n  out: {back:?}");
        }
    }

    #[test]
    fn encode_produces_reasonable_token_counts() {
        let Some(p) = fixture_path() else { eprintln!("skip"); return };
        let g = GgufFile::open(&p).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        // A short English sentence should tokenize to a handful of
        // tokens — far fewer than its byte length (BPE is doing its job).
        let text = "The quick brown fox jumps over the lazy dog.";
        let ids = tok.encode(text);
        eprintln!("encode({text:?}) = {} tokens: {ids:?}", ids.len());
        assert!(ids.len() < text.len() / 2,
            "expected BPE to compress; got {} tokens for {} bytes",
            ids.len(), text.len());
        assert!(!ids.is_empty());
    }
}
