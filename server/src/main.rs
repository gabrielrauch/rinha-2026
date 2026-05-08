use std::sync::Arc;

// SO_REUSEPORT-based multi-runtime: spawn N monoio runtimes (one per OS thread), each
// independently accepting on the SAME local port. The kernel hashes incoming connections
// across runtimes, giving us true parallelism with no shared mutex on the accept queue.
//
// N is chosen from RINHA_THREADS env var, falling back to 2 (matches our per-instance CPU
// budget of ~0.475 with one core spare for the kernel/networking).

const DEFAULT_THREADS: usize = 2;

fn main() -> anyhow::Result<()> {
    let blob_path = std::env::var("BLOB_PATH").unwrap_or_else(|_| "/index/blob.bin".into());
    let bind_str = std::env::var("BIND").unwrap_or_else(|_| "0.0.0.0:8000".into());
    let threads: usize = std::env::var("RINHA_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THREADS);

    let blob = Arc::new(server::blob::Blob::open(blob_path.as_ref())?);
    eprintln!(
        "loaded blob: {} vectors, hnsw layers={} entry={} M0={} M={}",
        blob.header().total_vectors,
        blob.header().hnsw_num_layers,
        blob.header().hnsw_entry_point,
        blob.header().hnsw_m0,
        blob.header().hnsw_m,
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

    // Create raw socket with SO_REUSEPORT so multiple runtimes can bind to the same port.
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

    // Build a monoio runtime for this OS thread.
    #[cfg(feature = "iouring")]
    let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .enable_timer()
        .build()
        .expect("monoio iouring runtime build");
    #[cfg(not(feature = "iouring"))]
    let mut rt = monoio::RuntimeBuilder::<monoio::LegacyDriver>::new()
        .enable_timer()
        .build()
        .expect("monoio legacy runtime build");

    rt.block_on(async move {
        let listener = monoio::net::TcpListener::from_std(std_listener)?;
        eprintln!("[rt-{tid}] listening on {bind_addr}");

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
        // unreachable, but typed:
        #[allow(unreachable_code)]
        Ok::<_, anyhow::Error>(())
    })
}
