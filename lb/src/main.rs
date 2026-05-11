use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use monoio::io::{AsyncReadRent, AsyncWriteRent, AsyncWriteRentExt, Splitable};
use monoio::net::{TcpListener, UnixStream};

// FusionDriver auto-picks IoUringDriver when the host kernel supports it (Mac
// Mini Haswell evaluator does), and falls back to LegacyDriver otherwise.
type Driver = monoio::FusionDriver;

fn main() -> anyhow::Result<()> {
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "9999".into())
        .parse()?;
    let upstreams: Vec<PathBuf> = std::env::var("UPSTREAMS")
        .unwrap_or_else(|_| "/run/sock/api1.sock,/run/sock/api2.sock".into())
        .split(',')
        .map(|s| PathBuf::from(s.trim()))
        .collect();
    anyhow::ensure!(!upstreams.is_empty(), "UPSTREAMS is empty");

    let upstreams = Arc::new(upstreams);
    let counter = Arc::new(AtomicUsize::new(0));

    let addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse()?;
    let std_listener = build_listener(addr)?;

    let mut rt = monoio::RuntimeBuilder::<Driver>::new()
        .enable_timer()
        .build()
        .expect("monoio runtime build");

    rt.block_on(async move {
        let listener = TcpListener::from_std(std_listener)?;
        eprintln!("lb listening on {addr} -> {} upstreams", upstreams.len());

        loop {
            let (client, _) = match listener.accept().await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("accept error: {e}");
                    continue;
                }
            };
            let _ = client.set_nodelay(true);

            let upstreams = upstreams.clone();
            let counter = counter.clone();

            monoio::spawn(async move {
                let idx = counter.fetch_add(1, Ordering::Relaxed) % upstreams.len();
                let upstream = match UnixStream::connect(&upstreams[idx]).await {
                    Ok(s) => s,
                    Err(_) => return,
                };

                let (c_r, c_w) = client.into_split();
                let (u_r, u_w) = upstream.into_split();

                let f1 = pipe(c_r, u_w);
                let f2 = pipe(u_r, c_w);
                let _ = monoio::join!(f1, f2);
            });
        }
        #[allow(unreachable_code)]
        Ok::<_, anyhow::Error>(())
    })
}

fn build_listener(addr: std::net::SocketAddr) -> anyhow::Result<std::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    let domain = if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let sock = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    sock.set_reuse_address(true)?;
    sock.set_nonblocking(true)?;
    sock.bind(&addr.into())?;
    sock.listen(4096)?;
    Ok(sock.into())
}

async fn pipe<R, W>(mut src: R, mut dst: W)
where
    R: AsyncReadRent,
    W: AsyncWriteRent + AsyncWriteRentExt,
{
    loop {
        let buf = vec![0u8; 8192];
        let (res, mut b) = src.read(buf).await;
        let n = match res {
            Ok(0) => return,
            Ok(n) => n,
            Err(_) => return,
        };
        b.truncate(n);
        let (res, _) = dst.write_all(b).await;
        if res.is_err() {
            return;
        }
    }
}
