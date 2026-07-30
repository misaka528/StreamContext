#!/usr/bin/env python3
"""Validate the bundled experiment arrays and their byte-level hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "Electricity.npz": {
        "sha256": "7cfa96c38aa8ba6d006cbaee9cb2e1be52c947ba9874b4c04d334921c65661bf",
        "x_shape": (45312, 14),
        "y_shape": (45312, 2),
    },
    "Sine.npz": {
        "sha256": "9f80a732a9251dcd103f6a27d0139026bfd9740df09934f715273c8544ee179c",
        "x_shape": (30000, 4),
        "y_shape": (30000, 2),
    },
}


def main() -> None:
    for name, expected in EXPECTED.items():
        path = ROOT / "data" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as payload:
            keys = set(payload.files)
            x_shape = payload["x_train"].shape
            y_shape = payload["y_train"].shape
        assert keys == {"x_train", "y_train"}, (name, keys)
        assert x_shape == expected["x_shape"], (name, x_shape)
        assert y_shape == expected["y_shape"], (name, y_shape)
        assert digest == expected["sha256"], (name, digest)
        print(f"OK  {name}  x={x_shape}  y={y_shape}  sha256={digest}")


if __name__ == "__main__":
    main()
