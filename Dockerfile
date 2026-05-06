# syntax=docker/dockerfile:1.7
FROM rust:1.84-bookworm AS builder
WORKDIR /work
COPY rust-toolchain.toml ./
COPY Cargo.toml Cargo.lock ./
COPY shared ./shared
COPY server ./server
COPY builder ./builder
COPY resources/normalization.json resources/mcc_risk.json resources/example-references.json resources/example-payloads.json ./resources/
ADD https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz ./resources/references.json.gz

# Build server (release)
RUN cargo build -p server --release

# Run offline builder to produce blob
RUN mkdir -p /out && cargo run -p builder --release -- ./resources /out/blob.bin

FROM gcr.io/distroless/cc-debian12 AS runtime
COPY --from=builder /work/target/release/server /server
COPY --from=builder /out/blob.bin /index/blob.bin
ENV BLOB_PATH=/index/blob.bin BIND=0.0.0.0:8000
EXPOSE 8000
ENTRYPOINT ["/server"]
