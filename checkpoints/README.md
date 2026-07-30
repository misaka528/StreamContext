# Model checkpoint

Place the compatible TabPFN v2.6 classifier checkpoint here as:

```text
tabpfn-v2.6-classifier-v2.6_default.ckpt
```

Expected SHA-256 for the checkpoint used in the reference runs:

```text
0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a
```

The weight file is excluded from Git because it is a third-party artifact with
separate distribution terms. You may instead pass any local path through
`--tabpfn-ckpt` or the `TABPFN_CKPT` environment variable.
