use crate::blob::Blob;
use crate::feature::vectorize_payload;
use crate::index::fraud_score;
use crate::payload::extract;
use crate::response;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::TcpStream;
use std::sync::Arc;

pub async fn handle_connection(blob: Arc<Blob>, mut stream: TcpStream) {
    let mut req_buf: Vec<u8> = Vec::with_capacity(4096);
    loop {
        let read_buf = vec![0u8; 4096];
        let (res, mut tmp) = stream.read(read_buf).await;
        let n = match res {
            Ok(0) => return,
            Ok(n) => n,
            Err(_) => return,
        };
        tmp.truncate(n);
        req_buf.extend_from_slice(&tmp);

        // Try to handle as many pipelined requests as we have full bytes for.
        loop {
            let outcome = try_handle_one(&blob, &mut stream, &req_buf).await;
            match outcome {
                HandleResult::NeedMore => break,
                HandleResult::Consumed(c) => {
                    req_buf.drain(..c);
                }
                HandleResult::Drop => return,
            }
            if req_buf.is_empty() {
                break;
            }
        }
    }
}

enum HandleResult {
    NeedMore,
    Consumed(usize),
    Drop,
}

async fn try_handle_one(
    blob: &Arc<Blob>,
    stream: &mut TcpStream,
    buf: &[u8],
) -> HandleResult {
    // Need full headers
    let header_end = match find_double_crlf(buf) {
        Some(p) => p,
        None => return HandleResult::NeedMore,
    };
    let request_line_end = match memchr_crlf(buf) {
        Some(p) => p,
        None => return HandleResult::NeedMore,
    };
    let request_line = &buf[..request_line_end];

    let (method, path) = match parse_request_line(request_line) {
        Some((m, p)) => (m, p),
        None => return HandleResult::Drop,
    };

    if method == b"GET" && path == b"/ready" {
        send_static(stream, response::RESP_READY).await;
        return HandleResult::Consumed(header_end);
    }

    if method == b"POST" && path == b"/fraud-score" {
        let cl = match content_length(&buf[..header_end]) {
            Some(c) => c,
            None => return HandleResult::Drop,
        };
        if buf.len() < header_end + cl {
            return HandleResult::NeedMore;
        }
        let body = &buf[header_end..header_end + cl];
        let resp = match extract(body) {
            Some(p) => match vectorize_payload(blob, &p) {
                Some(v) => {
                    let count = fraud_score(blob, &v);
                    response::for_count(count)
                }
                None => response::RESP_APPROVED_S0,
            },
            None => response::RESP_APPROVED_S0,
        };
        send_static(stream, resp).await;
        return HandleResult::Consumed(header_end + cl);
    }

    send_static(stream, response::RESP_NOT_FOUND).await;
    HandleResult::Consumed(header_end)
}

async fn send_static(stream: &mut TcpStream, payload: &'static [u8]) {
    // monoio's IoBuf is impl'd for &'static [u8], so we can hand the static slice directly
    // — zero allocation per response.
    let _ = stream.write_all(payload).await;
}

fn parse_request_line(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let sp1 = line.iter().position(|&c| c == b' ')?;
    let rest = &line[sp1 + 1..];
    let sp2 = rest.iter().position(|&c| c == b' ')?;
    Some((&line[..sp1], &rest[..sp2]))
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

fn memchr_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\r\n")
}

fn content_length(headers: &[u8]) -> Option<usize> {
    let s = std::str::from_utf8(headers).ok()?;
    for line in s.split("\r\n") {
        if let Some(rest) = line
            .strip_prefix("Content-Length:")
            .or_else(|| line.strip_prefix("content-length:"))
            .or_else(|| line.strip_prefix("Content-length:"))
        {
            return rest.trim().parse().ok();
        }
    }
    None
}
