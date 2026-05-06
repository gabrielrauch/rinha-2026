# syntax=docker/dockerfile:1.7
FROM rust:1.84-bookworm AS builder
WORKDIR /work
# Don't copy rust-toolchain.toml — that would force rustup to redownload all components.
# The base image already has Rust 1.84 installed.
COPY Cargo.toml Cargo.lock ./

# --- offline blob build: depends only on shared, builder, and resources.
# Server changes do NOT invalidate this stage's cache, so iterating on the server
# does not pay the k-means cost again.
COPY shared ./shared
COPY builder ./builder
COPY resources/normalization.json resources/mcc_risk.json resources/example-references.json resources/example-payloads.json ./resources/
ADD https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz ./resources/references.json.gz

# Stub out the server crate so the workspace is buildable without copying its real source.
RUN mkdir -p server/src && \
    printf '[package]\nname = "server"\nversion = "0.1.0"\nedition = "2021"\npublish = false\n[dependencies]\n' > server/Cargo.toml && \
    printf 'fn main() {}\n' > server/src/main.rs

RUN mkdir -p /out && cargo run -p builder --release -- ./resources /out/blob.bin

# --- server build: invalidates only on server source changes.
COPY server ./server
RUN cargo build -p server --release

FROM gcr.io/distroless/cc-debian12 AS runtime
COPY --from=builder /work/target/release/server /server
COPY --from=builder /out/blob.bin /index/blob.bin
ENV BLOB_PATH=/index/blob.bin BIND=0.0.0.0:8000
EXPOSE 8000
ENTRYPOINT ["/server"]
