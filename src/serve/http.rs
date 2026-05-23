//! Minimal blocking HTTP/1.1 — one request per connection, JSON bodies.
//! Just enough to host the OpenAI-shaped endpoints; no dependency.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: String,
}

/// Cap on the request body the server will read into memory. Anything
/// larger gets rejected with a 400 — a single overlong prompt past
/// this is way beyond any sensible context window, and unbounded
/// `Content-Length` is a trivial OOM vector.
pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;       // 8 MB

/// Cap on a single header line. Defends against slow-loris-style header
/// floods that would otherwise tie up a worker thread.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;           // 16 KB
pub const MAX_HEADERS: usize = 64;

/// Read one HTTP request — request line, headers, and the
/// Content-Length body. The connection is treated as single-shot.
///
/// Sets a 60-second read timeout on the underlying socket so a
/// misbehaving (or hostile) client can't hold a worker thread open
/// indefinitely. Rejects bodies above `MAX_BODY_BYTES` and headers
/// above `MAX_HEADER_BYTES` / `MAX_HEADERS`.
pub fn read_request(stream: &TcpStream) -> Result<Request, String> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(60)));

    let mut r = BufReader::new(stream);

    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| e.to_string())?;
    if line.len() > MAX_HEADER_BYTES {
        return Err(format!("request line too long ({} > {} bytes)",
                           line.len(), MAX_HEADER_BYTES));
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request line")?.to_string();
    let path   = parts.next().ok_or("missing request path")?.to_string();

    let mut content_length: usize = 0;
    let mut n_headers = 0;
    loop {
        if n_headers >= MAX_HEADERS {
            return Err(format!("too many headers (> {MAX_HEADERS})"));
        }
        let mut h = String::new();
        let n = r.read_line(&mut h).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        if h.len() > MAX_HEADER_BYTES {
            return Err(format!("header line too long ({} > {} bytes)",
                               h.len(), MAX_HEADER_BYTES));
        }
        let t = h.trim_end();
        if t.is_empty() { break; }              // blank line ends headers
        n_headers += 1;
        let lower = t.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Err(format!("Content-Length {} exceeds server cap ({} bytes)",
                           content_length, MAX_BODY_BYTES));
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
