#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${TABPFN_SRC:?Set TABPFN_SRC to the compatible TabPFN src directory}"
: "${TABPFN_CKPT:?Set TABPFN_CKPT to the v2.6 classifier checkpoint}"

python -u src/run_stream.py \
  --data data/Sine.npz \
  --dataset sine \
  --out-dir runs/sine_full_b16_c1024 \
  --streamcontext-mode full \
  --passive-memory priority \
  --batch-size 16 \
  --context-size 1024 \
  --confirmation-samples 64 \
  --full-routing-anchor-size 1024 \
  --full-recurrence-policy reuse \
  --full-quarantine-abrupt-evidence 1 \
  --tabpfn-src "$TABPFN_SRC" \
  --tabpfn-ckpt "$TABPFN_CKPT" \
  --tabpfn-device auto \
  --seeds 0 1 2
