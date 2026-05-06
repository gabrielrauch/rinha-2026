use std::sync::Arc;

// Runtime driver: use IORING when explicitly enabled, otherwise the portable legacy (epoll/kqueue)
// driver. The libkrun VM that podman uses on macOS does not support io_uring (returns ENOSYS),
// so the default has to be legacy. For the production Rinha submission set RINHA_IOURING=1 in
// the Dockerfile to opt back into io_uring on the test machine.
#[cfg_attr(feature = "iouring", monoio::main(driver = "iouring"))]
#[cfg_attr(not(feature = "iouring"), monoio::main(driver = "legacy"))]
async fn main() -> anyhow::Result<()> {
    let blob_path = std::env::var("BLOB_PATH").unwrap_or_else(|_| "/index/blob.bin".into());
    let bind = std::env::var("BIND").unwrap_or_else(|_| "0.0.0.0:8000".into());

    let blob = Arc::new(server::blob::Blob::open(blob_path.as_ref())?);
    eprintln!(
        "loaded blob: {} centroids, {} vectors",
        blob.header().num_centroids,
        blob.header().total_vectors
    );

    let listener = monoio::net::TcpListener::bind(&bind)?;
    eprintln!("listening on {}", bind);

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
}
