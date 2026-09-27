"""Inférence TFLite légère, partagée par l'export et la démo."""

import numpy as np
import tensorflow as tf

NORM_CLIP = 8.0


def normalize(X, mean, std, clip=NORM_CLIP):
    return np.clip((X - mean) / std, -clip, clip).astype(np.float32)


class TFLiteModel:
    """Interpréteur TFLite ; `norm=(mean, std)` si la normalisation est externe
    (cas du modèle int8, voir export_tflite.py)."""

    def __init__(self, content=None, path=None, norm=None):
        self.interp = tf.lite.Interpreter(model_content=content, model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.norm = norm

    def _quant(self, x):
        if self.norm is not None:
            x = normalize(x, *self.norm)
        scale, zero = self.inp["quantization"]
        if self.inp["dtype"] == np.int8:
            x = np.clip(np.round(x / scale + zero), -128, 127)
        return x.astype(self.inp["dtype"])

    def _dequant(self, y):
        scale, zero = self.out["quantization"]
        if self.out["dtype"] == np.int8:
            return (y.astype(np.float32) - zero) * scale
        return y

    def predict_one(self, x):
        self.interp.set_tensor(self.inp["index"], self._quant(x[None]))
        self.interp.invoke()
        return self._dequant(self.interp.get_tensor(self.out["index"]))[0]

    def predict(self, X):
        return np.stack([self.predict_one(x) for x in X])

    def largest_tensor_bytes(self):
        return max(
            int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
            for t in self.interp.get_tensor_details()
            if len(t["shape"]) > 1 and t["shape"][0] == 1
        )
