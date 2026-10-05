"""Feature extractors ("embedders") used by the bento verifier.

The pipeline only depends on the tiny `Embedder` interface, so the same logic runs with:
  * HistEmbedder   - pure OpenCV color histograms. No training, no model files. Good baseline.
  * OnnxEmbedder   - any ONNX backbone (e.g. MobileNet without its classifier head), CPU on a laptop.
  * HailoEmbedder  - placeholder for running a compiled model on the ASUS UGen300 (Hailo-10H).
"""
from __future__ import annotations
import cv2
import numpy as np


class HistEmbedder:
    """HSV histogram on a coarse spatial grid, Hellinger-normalised. Distance = Euclidean."""
    name = "hsv-hist"

    def __init__(self, size: int = 64, grid: int = 2, bins=(12, 4, 4)):
        self.size, self.grid, self.bins = size, grid, tuple(bins)

    def _hist(self, cell: np.ndarray) -> np.ndarray:
        h = cv2.calcHist([cell], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256]).flatten()
        return h / (h.sum() + 1e-9)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Soft binning: average the histogram with one whose bin edges are shifted by half a bin, so a
        # colour sitting right on a bin edge does not flip class under tiny brightness changes.
        shifted = hsv.astype(np.int16) + np.array([90 // self.bins[0], 128 // self.bins[1],
                                                   128 // self.bins[2]], np.int16)
        shifted[..., 0] %= 180
        shifted = np.clip(shifted, 0, 255).astype(np.uint8)
        step = self.size // self.grid
        feats = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                sl = (slice(gy * step, (gy + 1) * step), slice(gx * step, (gx + 1) * step))
                h = 0.5 * (self._hist(hsv[sl]) + self._hist(shifted[sl]))
                feats.append(np.sqrt(h))
        v = np.concatenate(feats)
        v /= np.linalg.norm(v) + 1e-9
        return v.astype(np.float32)


class OnnxEmbedder:
    """Neural embedding from an ONNX backbone. Pass a model that outputs a feature vector."""
    name = "onnx"

    def __init__(self, model_path: str, size: int = 224, providers=None):
        import onnxruntime as ort  # optional dependency
        self.sess = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        out = self.sess.run(None, {self.inp: x.transpose(2, 0, 1)[None]})[0].reshape(-1)
        return (out / (np.linalg.norm(out) + 1e-9)).astype(np.float32)


class HailoEmbedder:
    """TODO (Stage II): run the same backbone on the UGen300 / Hailo-10H.

    Typical flow: export the backbone to ONNX -> compile to a Hailo executable with the vendor
    toolchain -> run it through the vendor runtime. Follow the contest's "Compute Platform" page
    and the ASUS x Hailo briefing for the exact toolchain and supported models.
    Keep `embed()` returning an L2-normalised vector so nothing else in the pipeline changes.
    """
    name = "hailo"

    def __init__(self, *a, **k):
        raise NotImplementedError("Implement with the Hailo runtime once the UGen300 is available.")


def make_embedder(name: str = "hsv-hist", **kw):
    if name == "hsv-hist":
        return HistEmbedder(**kw)
    if name == "onnx":
        return OnnxEmbedder(**kw)
    if name == "hailo":
        return HailoEmbedder(**kw)
    raise ValueError(f"unknown embedder: {name}")
