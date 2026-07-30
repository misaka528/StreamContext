# StreamContext-TabPFN

Adaptive context management for frozen TabPFN on evolving tabular data streams.
The repository contains the complete stream-side implementation used in the
SINE and Electricity experiments. It does **not** contain GCN code: prediction,
change signals, state matching, and context updates are all driven by a frozen
TabPFN model.

## Method

The experiment follows a strict prequential **test-then-update** protocol. For
each arriving batch, the model first predicts with labels from past samples
only; the current labels are revealed only after evaluation.

The full method combines:

1. **Dual-timescale change detection.** A short/long-window label-loss signal is
   combined with normalized predictive entropy and changes in TabPFN query
   embeddings to distinguish gradual warnings from abrupt changes.
2. **Passive priority memory.** Within an active state, the fixed-size context
   is selected using recency, local representativeness, and class balance. The
   default score weights are 0.55, 0.35, and 0.10, respectively.
3. **Quarantined active arbitration.** After an abrupt event, the outgoing state
   is frozen and the next 64 labeled samples form an isolated recent window.
   If no historical concept matches, those samples initialize a new state. If a
   state matches, its stored context is merged with the recent window, then the
   same state is reactivated and continues updating. Confirmation samples are
   committed exactly once and never contaminate the outgoing state.
4. **Independent routing anchors.** Each state keeps a FIFO concept anchor for
   matching, while prediction uses the adaptive priority memory. Dormant-state
   time is excluded from the reactivated state's recency clock.

Available comparison modes are `sliding_window`, `passive_only`, `full`,
`hybrid_full`, and the earlier `hard_switch` implementation.

## Repository layout

```text
streamcontext-tabpfn/
├── data/
│   ├── Electricity.npz
│   ├── Sine.npz
│   ├── README.md
│   └── SHA256SUMS
├── results/reference_results.json
├── scripts/
│   ├── run_electricity.sh
│   ├── run_sine.sh
│   └── verify_data.py
├── src/
│   ├── dchr_tabpfn.py
│   ├── run_stream.py
│   └── tabpfn_extractor.py
├── .gitignore
├── environment.yml
└── requirements.txt
```

## Environment

The reference server used Python 3.10.18, NumPy 1.23.5, PyTorch 2.6.0+cu124,
Matplotlib 3.10.0, and scikit-learn 1.6.1. A CUDA GPU is strongly recommended,
especially for Electricity with batch size 1.

```bash
conda env create -f environment.yml
conda activate streamcontext-tabpfn
```

TabPFN is an external dependency and is not vendored here. The tested source
snapshot reports project version 7.1.1 and provides
`tabpfn.architectures.tabpfn_v2_6`. Install a compatible TabPFN source tree and
obtain the v2.6 classifier checkpoint, then either pass their locations on the
command line or export:

```bash
export TABPFN_SRC=/path/to/TabPFN/src
export TABPFN_CKPT=/path/to/tabpfn-v2.6-classifier-v2.6_default.ckpt
```

The reference checkpoint SHA-256 is:

```text
0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a
```

The checkpoint is intentionally excluded because model weights have their own
distribution terms. Review the upstream [TabPFN repository](https://github.com/PriorLabs/TabPFN)
and license before redistribution.

## Data

The exact NPZ arrays used in the experiments are included. Validate them before
running:

```bash
python scripts/verify_data.py
```

Both archives contain `x_train` and one-hot `y_train`; their shapes and hashes
are documented in `data/README.md`. The archives themselves do not embed their
original source or license metadata. Confirm upstream redistribution rights and
add the appropriate dataset citation/license before making a public release.

## Reproduce SINE

```bash
bash scripts/run_sine.sh
```

Equivalent explicit command:

```bash
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
  --tabpfn-device auto \
  --seeds 0 1 2
```

## Reproduce Electricity

```bash
bash scripts/run_electricity.sh
```

The reference Electricity setting uses batch size 1, context size 1500, and
seed 42. It is much slower than SINE because it performs one TabPFN forward pass
per arriving sample (45,312 passes rather than about 1,875 batched passes).

## Reference results

| Dataset | Setting | Prequential accuracy | Detection |
|---|---|---:|---|
| SINE | full, batch 16, context 1024, seeds 0/1/2 | **0.9344 ± 0.0000** | recall 1.0, 0 false positives, mean delay 11.2 samples |
| Electricity | full, batch 1, context 1500, seed 42 | **0.941472** | no point-wise ground-truth drift annotation |

Both results were obtained with the complete `full + priority` method and the
dataset-specific settings above. During the Electricity run, all detected
fluctuations were handled without a structural state transition; consequently,
the abrupt quarantine/reuse branch was not needed on that stream. See
`results/reference_results.json` for machine-readable details.

Each run writes:

- `result_seed*.json`: aggregate metrics, events, state statistics, and config;
- `batches_seed*.csv`: per-batch predictions and diagnostics;
- `accuracy_seed*.png`: rolling accuracy and detected/true change markers;
- `summary.json`: mean/std and event statistics across seeds.

## Earlier repository artifacts

The root-level `tabpfn_v2_6.py` and `*_realtime_acc*.csv` files were already in
this repository and are retained for provenance. That architecture file belongs
to the earlier soft-prompt experiment and is not imported by the implementation
under `src/`. Its original setup note is archived in
`legacy/README_soft_prompt.md`.

## Notes for a public GitHub release

- Add the paper title, authors, citation, and a project license once finalized.
- Do not commit the TabPFN checkpoint unless its license explicitly permits it.
- Preserve the dataset source/citation and redistribution license alongside the
  included NPZ files.
