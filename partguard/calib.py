"""Lighting calibration with a neutral gray reference card fixed somewhere in the camera frame.

Real kitchens change light all day (windows, lamps, camera auto-exposure). A gray card in view lets us
rescale each frame so 'gray' always reads the same, which keeps colour-based checks stable.
"""
from __future__ import annotations
import numpy as np


def roi_px(img, roi_norm):
    h, w = img.shape[:2]
    x, y, rw, rh = roi_norm
    return int(x * w), int(y * h), int(rw * w), int(rh * h)


def apply_card(img: np.ndarray, roi_norm, target: float = 128.0) -> np.ndarray:
    """Scale each colour channel so the card region averages `target`."""
    x, y, w, h = roi_px(img, roi_norm)
    mx, my = int(w * 0.1), int(h * 0.1)                       # ignore the card's edges
    patch = img[y + my:y + h - my, x + mx:x + w - mx].reshape(-1, 3).astype(np.float32)
    gain = target / np.maximum(patch.mean(axis=0), 1.0)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def overlaps(rect_a, rect_b) -> bool:
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
