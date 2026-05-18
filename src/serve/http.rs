//! Minimal blocking HTTP/1.1 — one request per connection, JSON bodies.
//! Just enough to host the OpenAI-shaped endpoints; no dependency.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: String,
}

/// Read one HTTP request — request line, headers, and the
/// Content-Length body. The connection is treated as single-shot.
pub fn read_request(stream: &TcpStream) -> Result<Request, String> {
    let mut r = BufReader::new(stream);

    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request line")?.to_string();
    let path   = parts.next().ok_or("missing request path")?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        let n = r.read_line(&mut h).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        let t = h.trim_end();
        if t.is_empty() { break; }              // blank line ends headers
        let lower = t.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    r.read_exact(&mut body).map_err(|e| e.to_string())?;
    let body = String::from_utf8(body).map_err(|_| "non-utf8 request body".to_string())?;

    Ok(Request { method, path, body })
}

/// Write a JSON response and close.
pub fn write_response(stream: &mut TcpStream, status: u16, status_text: &str, body: &str)
    -> std::io::Result<()>
{
    let resp = format!(
        "HTTP/1.1 {status} {status_text}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n{body}",
        body.len());
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}
