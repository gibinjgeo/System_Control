import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
import tensorflow as tf

assets = Path("assets")
models = ["keras_model.h5", "keras_model1.h5", "keras_model2.h5"]

for m in models:
    p = assets / m
    model = tf.keras.models.load_model(str(p), compile=False)
    out = p.with_suffix(".keras")
    model.save(out)
    print("Saved:", out)
