# rinha-2026-rust

Rust entry for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026). Fraud detection with exact k-NN over 3M reference vectors (14 dims) under 350MB / 1 CPU / Haswell.

**Score: 6000.00** (rinha cap) — p99 0.98ms, 0% failures, 0 wrong detections out of 54,100.

## Architecture

```
┌──────────────┐  TCP :9999  ┌──────┐  UDS + SCM_RIGHTS  ┌────────────────┐
│ rinha tester │ ──────────▶ │  lb  │ ─── fd handoff ───▶│ api1 / api2    │
└──────────────┘             └──────┘                    │ (KD-tree k-NN) │
                                                         └────────────────┘
```

### LB (`lb/`)

Single binary, sync libc, ~300 lines. For each accepted TCP socket:

1. Round-robin pick an upstream UDS path.
2. `sendmsg(SCM_RIGHTS)` the client FD across the persistent UDS connection.
3. Close our local FD; the kernel hands the live socket to the API process.

`WORKERS=2` threads share the listener via `SO_REUSEPORT` — kernel load-balances `accept` across them. Each worker holds its own persistent UDS FD per upstream and reconnects once on send failure.

The LB never touches request/response bytes.

### API (`server/`)

`monoio` runtime (io_uring / legacy fallback via `FusionDriver`). On startup:

1. `mmap` the blob (~92MB), `madvise(HUGEPAGE | RANDOM | WILLNEED)` on the hot regions, walk every page once to materialise it.
2. Open the `RINHA_FD_SOCK` Unix listener.
3. Spawn an OS accept thread → per-control-connection `recvmsg` threads extract FDs from SCM_RIGHTS messages and push them onto an mpsc channel.
4. The monoio task wakes via a socketpair byte, drains the channel, wraps each FD as `monoio::net::TcpStream`, and spawns the HTTP handler.

Hot path per request: HTTP parse (positional) → vectorise payload → KD-tree search → static response. mimalloc as the global allocator on Linux.

### Index (`builder/` + `server/src/knn.rs`)

Blob v4 format: **partitioned KD-tree with axis-aligned bounding-box pruning** + per-cluster i16 SoA blocks for SIMD scan.

- `partition_key` over discrete query features (sentinels in dim 5, MCC risk bucket, online/card-present/unknown-merchant flags, amount-vs-avg and tx-count bits) selects 1 of ≤256 partitions. Each partition has its own KD-tree.
- Each node carries `[min, max]` bboxes across all 14 dims. `lower_bound_vec(query, min, max)` is the closest a vector inside the node can possibly be — when that's already ≥ our 5th-best, we skip the subtree.
- Leaves (`LEAF_SIZE=128`, the MXLange-tuned value for this dataset) hold 8-lane SoA blocks. `scan_leaf` does AVX2 i32 squared L2 (`mullo_epi32` over `cvtepi16_epi32`'d lanes, accumulated as i64 to dodge overflow) and inserts into the top-5 by insertion sort.
- Global early termination: when the 5th-best squared distance falls below `EARLY_DISTANCE_LIMIT = (SCALE × 0.14)² = 1.96M`, we stop searching entirely. The 5 are already so close that no later visit could displace them.

Search flow per query:

1. `partition_key(query)` → primary partition → DFS its KD-tree with pruning. Most queries (~94% based on MXLange's numbers) terminate here.
2. Otherwise, compute lower bounds to every other partition, sort ascending, probe in order, break as soon as a partition's lower bound exceeds the current 5th-best.

Algorithm and on-disk format mirror MXLange's [`c-api-rinha2026`](https://github.com/MXLange/c-api-rinha2026) — credit where due. The magic header is `GOKNN001` for that reason.

## Repo layout

```
shared/   constants, BlobHeader, partition_key, lower_bound_vec, vectorize
builder/  blob construction (quantize references → KD-tree → write blob)
server/   monoio API: mmap blob, fd_listen, knn::fraud_count, HTTP wire
lb/       sync libc LB: accept TCP, SCM_RIGHTS the FD to api1/api2
```

## Building

CI publishes two images on every push to main:

- `ghcr.io/gabrielrauch/rinha-2026:<sha>` — the API
- `ghcr.io/gabrielrauch/rinha-2026-lb:<sha>` — the LB

Local sanity:

```sh
cargo build --release
# build the blob from references.json.gz (a few hundred ms on M1)
target/release/builder ./resources ./tmp/blob.bin 128
# run the api on TCP for local poking
BLOB_PATH=./tmp/blob.bin BIND=127.0.0.1:18000 target/release/server
curl -sf -m 2 -X POST http://localhost:18000/fraud-score -d @resources/example-payloads.json
```

## Environment variables

| Var | Default | Where |
|---|---|---|
| `RINHA_FD_SOCK` | _(unset)_ | server: path of UDS to receive FDs on (preferred) |
| `RINHA_SOCK` | _(unset)_ | server: legacy UDS byte-proxy mode (fallback) |
| `BIND` | `0.0.0.0:8000` | server: TCP bind (when neither UDS is set) |
| `BLOB_PATH` | `/index/blob.bin` | server: blob location |
| `PORT` | `9999` | lb: TCP listen port |
| `FD_UPSTREAMS` | `/run/sock/api1.sock,/run/sock/api2.sock` | lb: comma list of UDS to pass FDs to |
| `WORKERS` | `2` | lb: accept-loop threads |
