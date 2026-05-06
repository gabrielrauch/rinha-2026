use std::sync::Arc;

#[cfg_attr(target_os = "linux", monoio::main(driver = "iouring"))]
#[cfg_attr(not(target_os = "linux"), monoio::main(driver = "legacy"))]
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
