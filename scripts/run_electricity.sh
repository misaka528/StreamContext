#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${TABPFN_SRC:?Set TABPFN_SRC to the compatible TabPFN src directory}"
: "${TABPFN_CKPT:?Set TABPFN_CKPT to the v2.6 classifier checkpoint}"

python -u src/run_stream.py \
  --data data/Electricity.npz \
  --dataset electricity \
  --out-dir runs/electricity_full_b1_c1500 \
  --streamcontext-mode full \
  --passive-memory priority \
  --batch-size 1 \
  --context-size 1500 \
  --confirmation-samples 64 \
  --full-routing-anchor-size 1024 \
  --full-recurrence-policy reuse \
  --full-quarantine-abrupt-evidence 1 \
  --tabpfn-src "$TABPFN_SRC" \
  --tabpfn-ckpt "$TABPFN_CKPT" \
  --tabpfn-device auto \
  --seeds 42
