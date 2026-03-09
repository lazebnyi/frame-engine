import time

import numpy as np
from PIL import Image

TARGET_W = 640
TARGET_H = 480


class Processor:
    """Preprocess raw frames and run model inference."""

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Resize each frame to 640×480 and convert to grayscale."""
        n = data.shape[0]
        out = np.empty((n, TARGET_H, TARGET_W), dtype=np.uint8)
        for i in range(n):
            img = Image.fromarray(data[i])
            img = img.convert("L")
            img = img.resize((TARGET_W, TARGET_H), Image.BILINEAR)
            out[i] = np.asarray(img)
        return out

    def run_inference(self, _data: np.ndarray) -> dict:
        """Simulated model forward-pass. Replace with real model call."""
        time.sleep(0.1)
        return {}
