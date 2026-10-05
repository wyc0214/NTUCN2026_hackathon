"""Bakery quality check: segment each item, measure size / shape / browning, give PASS/FAIL + reasons.

Pure OpenCV, no training data needed. Assumes a fixed camera and a plain, contrasting tray/belt.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import cv2
import numpy as np

from .calib import apply_card, roi_px, overlaps


@dataclass
class ItemResult:
    id: int
    bbox: tuple              # x, y, w, h (pixels)
    diameter_px: float       # equivalent diameter
    diameter_mm: float | None
    roundness: float         # minor/major axis of fitted ellipse (1.0 = round)
    solidity: float          # area / convex-hull area (low = dents or broken edges)
    mean_L: float            # lightness 0-100 (lower = darker)
    std_L: float
    burnt_ratio: float
    passed: bool
    reasons: list = field(default_factory=list)   # reason codes, see report.py


def _lab_float(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0     # OpenCV stores L in 0-255
    return lab


def foreground_mask(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Return contours of items that differ from the background (estimated from the image border)."""
    h, w = img_bgr.shape[:2]
    lab = _lab_float(img_bgr)
    b = max(2, int(0.03 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    thr = cfg["seg"]["bg_distance"]
    if thr == "otsu":
        _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(d8, float(thr) * 4, 255, cv2.THRESH_BINARY)
    k = max(3, int(cfg["seg"].get("morph_frac", 0.01) * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment(img_bgr: np.ndarray, cfg: dict) -> list:
    """Return contours of items that differ from the background (estimated from the image border)."""
    mask = foreground_mask(img_bgr, cfg)
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg["seg"]["min_area_frac"] * h * w
    return [c for c in cnts if cv2.contourArea(c) >= min_area]


def inspect(img_bgr: np.ndarray, cfg: dict) -> list:
    cfg = cfg["bakery"] if "bakery" in cfg else cfg
    cal = cfg.get("calibration")
    if cal:
        img_bgr = apply_card(img_bgr, cal["card_roi"], cal.get("target_gray", 128))
    lab = _lab_float(img_bgr)
    L = lab[..., 0]
    cnts = segment(img_bgr, cfg)
    if cal:
        card = roi_px(img_bgr, cal["card_roi"])
        cnts = [c for c in cnts if not overlaps(cv2.boundingRect(c), card)]

    def centre(c):
        x, y, w, h = cv2.boundingRect(c)
        return (y + h / 2) // 80, x + w / 2
    cnts = sorted(cnts, key=centre)

    diams = [math.sqrt(4 * cv2.contourArea(c) / math.pi) for c in cnts]
    px_per_mm = cfg.get("px_per_mm")
    med = float(np.median(diams)) if diams else 0.0

    results = []
    for i, (c, d_px) in enumerate(zip(cnts, diams), start=1):
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / (hull_area + 1e-9)
        if len(c) >= 5:
            (_, _), (ax1, ax2), _ = cv2.fitEllipse(c)
            roundness = min(ax1, ax2) / (max(ax1, ax2) + 1e-9)
        else:
            roundness = 1.0

        # work on the item's bounding box only (much faster than full-frame masks)
        x, y, w, h = cv2.boundingRect(c)
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c - np.array([x, y])], -1, 255, -1)
        er = max(3, int(0.12 * d_px)) | 1
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
        if inner.sum() == 0:
            inner = mask
        vals = L[y:y + h, x:x + w][inner > 0]
        mean_L, std_L = float(vals.mean()), float(vals.std())
        burnt = float((vals < cfg["color"]["burnt_L"]).mean())

        reasons = []
        if px_per_mm:
            d_mm = d_px / px_per_mm
            if d_mm < cfg["size"]["min_mm"]:
                reasons.append("SIZE_SMALL")
            elif d_mm > cfg["size"]["max_mm"]:
                reasons.append("SIZE_LARGE")
        else:
            d_mm = None
            tol = cfg["size"]["rel_tol"]
            if d_px < med * (1 - tol):
                reasons.append("SIZE_SMALL")
            elif d_px > med * (1 + tol):
                reasons.append("SIZE_LARGE")
        shp = cfg.get("shape") or {}
        if (shp.get("min_roundness") and roundness < shp["min_roundness"]) or \
           (shp.get("min_solidity") and solidity < shp["min_solidity"]):
            reasons.append("DEFORMED")
        if mean_L > cfg["color"]["L_max"]:
            reasons.append("UNDERBAKED")
        elif mean_L < cfg["color"]["L_min"]:
            reasons.append("OVERBAKED")
        if burnt > cfg["color"]["max_burnt_ratio"]:
            reasons.append("BURNT_SPOTS")
        if std_L > cfg["color"]["max_std_L"]:
            reasons.append("UNEVEN_COLOR")

        results.append(ItemResult(i, (x, y, w, h), d_px, d_mm, roundness, solidity,
                                  mean_L, std_L, burnt, not reasons, reasons))
    return results
