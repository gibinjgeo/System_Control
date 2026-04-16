from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Must be set before importing tensorflow
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

# Tune CPU threading for Ryzen 7 5800H (8 cores / 16 threads).
# intra: parallelism within a single op (e.g. matrix multiply)
# inter: parallelism across independent ops in a graph
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)


def _load_model(model_path: Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(str(model_path), compile=False)


class TMClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        self.model = _load_model(Path(model_path))
        self.id_to_label = self._read_labels(Path(labels_path))

    @staticmethod
    def _read_labels(p: Path) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            idx = int(parts[0])
            name = parts[1] if len(parts) > 1 else ""
            mapping[idx] = name
        return mapping

    def predict(self, x: np.ndarray) -> Tuple[int, str, float]:
        # Call the model directly (avoids the logging overhead of model.predict()
        # which is optimised for batched dataset inference, not single samples)
        pred = self.model(x, training=False).numpy()[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx])
        label = self.id_to_label.get(idx, f"ID_{idx}")
        return idx, label, conf
