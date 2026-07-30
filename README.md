# StreamContext-TabPFN

Adaptive context management for frozen TabPFN on evolving tabular data streams.
The repository contains the complete stream-side implementation used in the
SINE and Electricity experiments. It does **not** contain GCN code: prediction,
change signals, state matching, and context updates are all driven by a frozen
TabPFN model.


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
Matplotlib 3.10.0, and scikit-learn 1.6.1.

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

The reference real-time dataset setting uses batch size 1, context size 1500, and
seed 42. It is much slower than SINE because it performs one TabPFN forward pass
per arriving sample (45,312 passes rather than about 1,875 batched passes).




## Earlier repository artifacts

The root-level `tabpfn_v2_6.py` and `*_realtime_acc*.csv` files were already in
this repository and are retained for provenance. That architecture file belongs
to the earlier soft-prompt experiment and is not imported by the implementation
under `src/`. Its original setup note is archived in
`legacy/README_soft_prompt.md`.

