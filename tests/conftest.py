import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

APP = ROOT / "app"
DEMO_KEYS = ("db2_s1", "db3_s8")


@pytest.fixture(scope="session")
def embedded():
    """Modèles int8 et extraits réels embarqués dans la démo (répétition de test)."""
    from inference import TFLiteModel

    out = {}
    for key in DEMO_KEYS:
        norm = json.loads((APP / "model" / f"cnn_{key}_norm.json").read_text())
        std = np.array(norm["std"], np.float32)
        model = TFLiteModel(
            path=str(APP / "model" / f"cnn_{key}_int8.tflite"),
            norm=(np.array(norm["mean"], np.float32), std),
        )
        d = np.load(APP / "demo_data" / f"{key}_rep2.npz")
        out[key] = {
            "model": model,
            "std": std,
            "emg": d["emg_uv"].astype(np.float32) * 1e-6,
            "labels": d["labels"].astype(int),
            "embarque": json.loads((APP / "model" / f"cnn_{key}_embarque.json").read_text()),
        }
    return out
