from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# MUST be set before importing tensorflow to use legacy tf.keras (Keras 2 via tf-keras)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# Optional: quiet TensorFlow logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402


def _load_model(model_path: Path):
    """
    Loads TeachableMachine/TF models saved as .keras or .h5.
    Your converted .keras files are legacy-tf-keras based, so we force legacy mode above.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # compile=False avoids optimizer/loss deserialization issues
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
        pred = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx])
        label = self.id_to_label.get(idx, f"ID_{idx}")
        return idx, label, conf
