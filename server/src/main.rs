use std::path::PathBuf;
use std::sync::Arc;

// Per-process listener: UDS when RINHA_SOCK is set (production path through HAProxy
// unix@... backends — eliminates TCP loopback overhead). Falls back to TCP for local
// `cargo run` without compose.

fn main() -> anyhow::Result<()> {
    let blob_path = std::env::var("BLOB_PATH").unwrap_or_else(|_| "/index/blob.bin".into());
    let blob = Arc::new(server::blob::Blob::open(blob_path.as_ref())?);
    eprintln!(
        "loaded blob: {} vectors, hnsw layers={} entry={} M0={} M={}",
        blob.header().total_vectors,
        blob.header().hnsw_num_layers,
        blob.header().hnsw_entry_point,
        blob.header().hnsw_m0,
        blob.header().hnsw_m,
    );

    if let Ok(sock_path) = std::env::var("RINHA_SOCK") {
        run_uds(blob, PathBuf::from(sock_path))
    } else {
        let bind_str = std::env::var("BIND").unwrap_or_else(|_| "0.0.0.0:8000".into());
        let bind_addr: std::net::SocketAddr = bind_str.parse()?;
        run_tcp(blob, bind_addr)
    }
}

// FusionDriver picks IoUringDriver at runtime when the kernel supports it
// (Mac Mini Haswell: yes), and falls back to LegacyDriver otherwise. This
// matters for UDS: monoio's LegacyDriver has a known quirk where
// UnixListener::accept doesn't wake reliably; IoUringDriver does.
type Driver = monoio::FusionDriver;

fn run_uds(blob: Arc<server::blob::Blob>, sock_path: PathBuf) -> anyhow::Result<()> {
    use monoio::net::{ListenerOpts, UnixListener};
    use std::os::unix::fs::PermissionsExt;

    if let Some(parent) = sock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(&sock_path); // ENOENT is fine

    let mut rt = monoio::RuntimeBuilder::<Driver>::new()
        .enable_timer()
        .build()
        .expect("monoio runtime build");
    let sp = sock_path.clone();
    rt.block_on(async move {
        let opts = ListenerOpts::new();
        let listener = UnixListener::bind_with_config(&sp, &opts)?;
        // HAProxy may connect as a different uid — try to make the socket world-RW.
        let _ = std::fs::set_permissions(&sp, std::fs::Permissions::from_mode(0o666));
        eprintln!("listening on uds {}", sp.display());

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let blob = blob.clone();
                    monoio::spawn(async move {
                        server::wire::handle_connection(blob, stream).await;
                    });
                }
                Err(e) => eprintln!("accept error: {e}"),
            }
        }
        #[allow(unreachable_code)]
        Ok::<_, anyhow::Error>(())
    })
}

fn run_tcp(blob: Arc<server::blob::Blob>, bind_addr: std::net::SocketAddr) -> anyhow::Result<()> {
    use socket2::{Domain, Protocol, Socket, Type};

    let domain = if bind_addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let sock = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    sock.set_reuse_address(true)?;
    sock.set_nodelay(true)?;
    sock.set_nonblocking(true)?;
    sock.bind(&bind_addr.into())?;
    sock.listen(1024)?;
    let std_listener: std::net::TcpListener = sock.into();

    let mut rt = monoio::RuntimeBuilder::<Driver>::new()
        .enable_timer()
        .build()
        .expect("monoio runtime build");
    rt.block_on(async move {
        let listener = monoio::net::TcpListener::from_std(std_listener)?;
        eprintln!("listening on tcp {bind_addr}");

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let blob = blob.clone();
                    monoio::spawn(async move {
                        server::wire::handle_connection(blob, stream).await;
                    });
                }
                Err(e) => eprintln!("accept error: {e}"),
            }
        }
        #[allow(unreachable_code)]
        Ok::<_, anyhow::Error>(())
    })
}
