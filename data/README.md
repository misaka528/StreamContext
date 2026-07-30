# Included experiment arrays

| File | Features | Labels | Shape | SHA-256 |
|---|---:|---:|---|---|
| `Sine.npz` | 4 | 2 | `x_train=(30000,4)`, `y_train=(30000,2)` | `9f80a732a9251dcd103f6a27d0139026bfd9740df09934f715273c8544ee179c` |
| `Electricity.npz` | 14 | 2 | `x_train=(45312,14)`, `y_train=(45312,2)` | `7cfa96c38aa8ba6d006cbaee9cb2e1be52c947ba9874b4c04d334921c65661bf` |

Both files are ordered streams and must not be shuffled. Labels are one-hot
encoded; the runner converts them to integer class indices. Electricity
features lie in `[0,1]`; neither archive contains NaN values.

SINE consists of six 5,000-sample segments with concept sequence
`1 → 2 → 1 → 2 → 1 → 2`, so its known changes are at positions 5,000, 10,000,
15,000, 20,000, and 25,000. Electricity is treated as a naturally evolving
stream and has no point-wise drift annotations in this repository.

The NPZ archives do not include upstream provenance or license metadata.
Before public redistribution, add the original source citation and verify that
the applicable dataset license permits bundling these derived arrays.
