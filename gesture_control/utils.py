import numpy as np
import cv2
from pathlib import Path

def pick_model(assets_dir: Path, stem: str):
    """
    Prefer the newer .keras file if it exists, else fall back to .h5.
    Example: stem='keras_model' -> keras_model.keras preferred over keras_model.h5
    """
    assets_dir = Path(assets_dir)
    p_keras = assets_dir / f"{stem}.keras"
    p_h5 = assets_dir / f"{stem}.h5"

    if p_keras.exists():
        return p_keras
    if p_h5.exists():
        return p_h5

    raise FileNotFoundError(f"No model found for '{stem}' in {assets_dir} (expected .keras or .h5)")


def clamp_bbox(x, y, w, h, img_w, img_h):
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h

def make_square_white(img_crop, out_size: int):
    """
    Places a crop into a white square canvas of size out_size x out_size,
    preserving aspect ratio (similar to what you were doing).
    """
    white = np.ones((out_size, out_size, 3), np.uint8) * 255
    h, w = img_crop.shape[:2]

    if h == 0 or w == 0:
        return white

    ar = h / w
    if ar > 1:
        scale = out_size / h
        new_w = int(w * scale)
        resized = cv2.resize(img_crop, (new_w, out_size))
        x0 = (out_size - new_w) // 2
        white[:, x0:x0 + new_w] = resized
    else:
        scale = out_size / w
        new_h = int(h * scale)
        resized = cv2.resize(img_crop, (out_size, new_h))
        y0 = (out_size - new_h) // 2
        white[y0:y0 + new_h, :] = resized

    return white

def preprocess_for_tm(img_224):
    """
    Teachable Machine style normalization:
    image = (image/127.5) - 1
    """
    x = img_224.astype(np.float32)
    x = x.reshape(1, 224, 224, 3)
    x = (x / 127.5) - 1.0
    return x
