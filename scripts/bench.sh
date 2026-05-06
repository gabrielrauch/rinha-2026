#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Clone the official repo to get the test harness
git clone --depth 1 https://github.com/zanfranceschi/rinha-de-backend-2026 "$WORK/rinha"

(
  cd "$ROOT"
  docker compose down --remove-orphans 2>/dev/null || true
  docker compose up -d
  for i in {1..20}; do
    if curl -sf http://127.0.0.1:9999/ready >/dev/null; then break; fi
    sleep 1
  done
)

(
  cd "$WORK/rinha"
  K6_NO_USAGE_REPORT=true k6 run test/test.js
)
