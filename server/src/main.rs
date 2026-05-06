use std::sync::Arc;

// SO_REUSEPORT-based multi-runtime: spawn N monoio runtimes (one per OS thread), each
// independently accepting on the SAME local port. The kernel hashes incoming connections
// across runtimes, giving us true parallelism with no shared mutex on the accept queue.
//
// Driver selection is automatic at runtime: try io_uring first (best on Linux 5.6+);
// fall back to legacy (epoll/kqueue) when io_uring is unavailable, e.g. inside libkrun
// VMs on macOS or kernels without the syscall.
//
// N is read from RINHA_THREADS env var, defaulting to 3 (slightly oversubscribes the
// 0.475 CPU/container budget on the Mac Mini test box, which has 4 SMT threads —
// while one thread is in syscall the others compute).

const DEFAULT_THREADS: usize = 3;

fn main() -> anyhow::Result<()> {
    let blob_path = std::env::var("BLOB_PATH").unwrap_or_else(|_| "/index/blob.bin".into());
    let bind_str = std::env::var("BIND").unwrap_or_else(|_| "0.0.0.0:8000".into());
    let threads: usize = std::env::var("RINHA_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THREADS);

    let blob = Arc::new(server::blob::Blob::open(blob_path.as_ref())?);
    eprintln!(
        "loaded blob: {} centroids, {} vectors",
        blob.header().num_centroids,
        blob.header().total_vectors
    );

    let bind_addr: std::net::SocketAddr = bind_str.parse()?;

    let mut handles = Vec::with_capacity(threads);
    for tid in 0..threads {
        let blob = blob.clone();
        let handle = std::thread::Builder::new()
            .name(format!("rinha-rt-{tid}"))
            .spawn(move || {
                if let Err(e) = run_runtime(tid, blob, bind_addr) {
                    eprintln!("runtime {tid} exited: {e}");
                }
            })?;
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn run_runtime(
    tid: usize,
    blob: Arc<server::blob::Blob>,
    bind_addr: std::net::SocketAddr,
) -> anyhow::Result<()> {
    use socket2::{Domain, Protocol, Socket, Type};

    let domain = if bind_addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let sock = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    sock.set_reuse_address(true)?;
    #[cfg(unix)]
    sock.set_reuse_port(true)?;
    sock.set_nodelay(true)?;
    sock.set_nonblocking(true)?;
    sock.bind(&bind_addr.into())?;
    sock.listen(1024)?;

    let std_listener: std::net::TcpListener = sock.into();

    // Try io_uring first on Linux; fall back to legacy if the runtime won't build
    // (e.g. inside libkrun, or on non-Linux dev boxes where IoUringDriver doesn't exist).
    #[cfg(target_os = "linux")]
    {
        match run_iouring(tid, &blob, &std_listener, bind_addr) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("[rt-{tid}] io_uring unavailable ({e}); falling back to legacy");
            }
        }
    }
    run_legacy(tid, blob, std_listener, bind_addr)
}

#[cfg(target_os = "linux")]
fn run_iouring(
    tid: usize,
    blob: &Arc<server::blob::Blob>,
    listener: &std::net::TcpListener,
    bind_addr: std::net::SocketAddr,
) -> anyhow::Result<()> {
    let listener = listener.try_clone()?;
    let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .enable_timer()
        .build()
        .map_err(|e| anyhow::anyhow!("io_uring runtime build failed: {e}"))?;
    let blob = blob.clone();
    rt.block_on(async move {
        let listener = monoio::net::TcpListener::from_std(listener)?;
        eprintln!("[rt-{tid}] iouring listening on {bind_addr}");
        accept_loop(tid, blob, listener).await;
        Ok::<_, anyhow::Error>(())
    })
}

fn run_legacy(
    tid: usize,
    blob: Arc<server::blob::Blob>,
    listener: std::net::TcpListener,
    bind_addr: std::net::SocketAddr,
) -> anyhow::Result<()> {
    let mut rt = monoio::RuntimeBuilder::<monoio::LegacyDriver>::new()
        .enable_timer()
        .build()
        .map_err(|e| anyhow::anyhow!("legacy runtime build failed: {e}"))?;
    rt.block_on(async move {
        let listener = monoio::net::TcpListener::from_std(listener)?;
        eprintln!("[rt-{tid}] legacy listening on {bind_addr}");
        accept_loop(tid, blob, listener).await;
        Ok::<_, anyhow::Error>(())
    })
}

async fn accept_loop(
    tid: usize,
    blob: Arc<server::blob::Blob>,
    listener: monoio::net::TcpListener,
) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let blob = blob.clone();
                monoio::spawn(async move {
                    server::wire::handle_connection(blob, stream).await;
                });
            }
            Err(e) => eprintln!("[rt-{tid}] accept error: {e}"),
        }
    }
}
