//! Minimal JSON — just enough to parse OpenAI-shaped requests and emit
//! responses. Recursive-descent parser, escaping emitter; no dependency.

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Object field lookup (returns the first match).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Json::Str(s) => Some(s.as_str()), _ => None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self { Json::Num(n) => Some(*n), _ => None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { Json::Bool(b) => Some(*b), _ => None }
    }
    #[allow(dead_code)]   // public API symmetry with as_f64/as_bool/as_str
    pub fn as_array(&self) -> Option<&[Json]> {
        match self { Json::Arr(a) => Some(a.as_slice()), _ => None }
    }

    /// Parse a complete JSON document.
    pub fn parse(s: &str) -> Result<Json, String> {
        let mut p = Parser { b: s.as_bytes(), i: 0, depth: 0 };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.i != p.b.len() {
            return Err("trailing data after JSON value".into());
        }
        Ok(v)
    }

    /// Serialise to a compact JSON string.
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.is_finite() && n.abs() < 1e15 {
                    out.push_str(&(*n as i64).to_string());
                } else {
                    out.push_str(&n.to_string());
                }
            }
            Json::Str(s) => write_escaped(s, out),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    write_escaped(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_escaped(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    /// Current nesting depth (object/array). Bumped on `{`/`[`, decremented
    /// on the matching `}`/`]`. Capped at `MAX_DEPTH` so a hostile client
    /// can't stack-overflow the recursive descent with `[[[[…]]]]`.
    depth: usize,
}

const MAX_DEPTH: usize = 256;

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len()
            && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        if self.i >= self.b.len() {
            return Err("unexpected end of JSON".into());
        }
        match self.b[self.i] {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => { self.lit("true")?;  Ok(Json::Bool(true)) }
            b'f' => { self.lit("false")?; Ok(Json::Bool(false)) }
            b'n' => { self.lit("null")?;  Ok(Json::Null) }
            b'-' | b'0'..=b'9' => self.number(),
            c => Err(format!("unexpected byte 0x{c:02x} in JSON")),
        }
    }

    fn lit(&mut self, s: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(())
        } else {
            Err(format!("expected '{s}'"))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i])
            .map_err(|_| "non-utf8 number".to_string())?;
        s.parse::<f64>().map(Json::Num).map_err(|_| format!("invalid number '{s}'"))
    }

    fn string(&mut self) -> Result<String, String> {
        debug_assert_eq!(self.b[self.i], b'"');
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out)
                    .map_err(|_| "non-utf8 string".to_string()),
                b'\\' => {
                    if self.i >= self.b.len() { return Err("dangling escape".into()); }
                    let e = self.b[self.i];
                    self.i += 1;
                    match e {
                        b'"'  => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/'  => out.push(b'/'),
                        b'n'  => out.push(b'\n'),
                        b'r'  => out.push(b'\r'),
                        b't'  => out.push(b'\t'),
                        b'b'  => out.push(0x08),
                        b'f'  => out.push(0x0c),
                        b'u'  => {
                            if self.i + 4 > self.b.len() { return Err("bad \\u".into()); }
                            let hex = std::str::from_utf8(&self.b[self.i..self.i + 4])
                                .map_err(|_| "bad \\u".to_string())?;
                            let cp = u32::from_str_radix(hex, 16)
                                .map_err(|_| "bad \\u".to_string())?;
                            self.i += 4;
                            let ch = char::from_u32(cp).unwrap_or('\u{fffd}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err("invalid escape".into()),
                    }
                }
                // Any other byte (incl. UTF-8 continuation bytes) is literal.
                _ => out.push(c),
            }
        }
        Err("unterminated string".into())
    }

    fn array(&mut self) -> Result<Json, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(format!("JSON nesting depth exceeds {MAX_DEPTH}"));
        }
        self.i += 1; // '['
        let mut a = Vec::new();
        self.skip_ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1; self.depth -= 1; return Ok(Json::Arr(a));
        }
        loop {
            a.push(self.value()?);
            self.skip_ws();
            match self.b.get(self.i) {
                Some(b',') => { self.i += 1; }
                Some(b']') => { self.i += 1; self.depth -= 1; return Ok(Json::Arr(a)); }
                _ => return Err("expected ',' or ']'".into()),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(format!("JSON nesting depth exceeds {MAX_DEPTH}"));
        }
        self.i += 1; // '{'
        let mut kv = Vec::new();
        self.skip_ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1; self.depth -= 1; return Ok(Json::Obj(kv));
        }
        loop {
            self.skip_ws();
            if self.b.get(self.i) != Some(&b'"') {
                return Err("expected object key string".into());
            }
            let key = self.string()?;
            self.skip_ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err("expected ':' after object key".into());
            }
            self.i += 1;
            let val = self.value()?;
            kv.push((key, val));
            self.skip_ws();
            match self.b.get(self.i) {
                Some(b',') => { self.i += 1; }
                Some(b'}') => { self.i += 1; self.depth -= 1; return Ok(Json::Obj(kv)); }
                _ => return Err("expected ',' or '}'".into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn round_trips_an_openai_request() {
        let src = r#"{"model":"big","prompt":"Hello\n\"world\"","max_tokens":64,
                      "temperature":0.7,"stream":false}"#;
        let j = Json::parse(src).expect("parse");
        assert_eq!(j.get("prompt").unwrap().as_str().unwrap(), "Hello\n\"world\"");
        assert_eq!(j.get("max_tokens").unwrap().as_f64().unwrap(), 64.0);
        assert_eq!(j.get("temperature").unwrap().as_f64().unwrap(), 0.7);
        // emit + re-parse is stable
        let s = j.to_string();
        let j2 = Json::parse(&s).expect("reparse");
        assert_eq!(j2.get("prompt").unwrap().as_str().unwrap(), "Hello\n\"world\"");
    }

    #[test]
    fn parses_nested_and_unicode() {
        let j = Json::parse(r#"{"messages":[{"role":"user","content":"café"}]}"#).unwrap();
        let m = &j.get("messages").unwrap().as_array().unwrap()[0];
        assert_eq!(m.get("content").unwrap().as_str().unwrap(), "café");
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(Json::parse("{} extra").is_err());
        assert!(Json::parse("{\"a\":}").is_err());
    }

    #[test]
    fn rejects_unbalanced_brackets() {
        assert!(Json::parse("[1, 2, 3").is_err());
        assert!(Json::parse("]").is_err());
        assert!(Json::parse("{\"a\": 1").is_err());
        assert!(Json::parse("}").is_err());
    }

    #[test]
    fn rejects_empty_and_whitespace() {
        assert!(Json::parse("").is_err());
        assert!(Json::parse("   \n\t  ").is_err());
    }

    #[test]
    fn rejects_too_deeply_nested() {
        // 300 levels of `[…]` — well past MAX_DEPTH=256. Without the
        // depth guard this would stack-overflow the recursive descent
        // and crash the serve worker.
        let depth = 300;
        let src: String = std::iter::repeat('[').take(depth)
            .chain(std::iter::repeat(']').take(depth)).collect();
        let r = Json::parse(&src);
        assert!(r.is_err(), "should reject {depth}-deep nesting, got {r:?}");
        assert!(r.unwrap_err().contains("depth"),
            "error should mention depth limit");
    }

    #[test]
    fn accepts_reasonably_nested() {
        // 200 levels — below MAX_DEPTH, should parse fine.
        let depth = 200;
        let src: String = std::iter::repeat('[').take(depth)
            .chain(std::iter::repeat(']').take(depth)).collect();
        assert!(Json::parse(&src).is_ok(),
            "should accept {depth}-deep nesting");
    }

    #[test]
    fn handles_escape_sequences_in_strings() {
        let j = Json::parse(r#""hello\nworld\t\"quoted\"""#).unwrap();
        assert_eq!(j.as_str().unwrap(), "hello\nworld\t\"quoted\"");
    }

    #[test]
    fn handles_unicode_escapes() {
        // é = é
        let j = Json::parse(r#""café""#).unwrap();
        assert_eq!(j.as_str().unwrap(), "café");
    }

    #[test]
    fn handles_scientific_notation_numbers() {
        let j = Json::parse("1.5e10").unwrap();
        assert!((j.as_f64().unwrap() - 1.5e10).abs() < 1e-3);
        let j = Json::parse("-2.5E-3").unwrap();
        assert!((j.as_f64().unwrap() + 2.5e-3).abs() < 1e-9);
    }

    #[test]
    fn empty_array_and_object() {
        assert!(matches!(Json::parse("[]").unwrap(), Json::Arr(ref v) if v.is_empty()));
        assert!(matches!(Json::parse("{}").unwrap(), Json::Obj(ref v) if v.is_empty()));
    }
}
