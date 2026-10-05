"""Metal-part inspection (washers, nuts, stamped rings): size, shape, hole, rust and scratches.

Pure OpenCV, no training. Assumes a fixed camera, a plain contrasting surface, and diffuse light.
Each part gets PASS/FAIL plus reason codes. Size and hole size are judged against the batch median
unless `px_per_mm` and drawing tolerances are provided.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import cv2
import numpy as np

from .bakery import foreground_mask, _lab_float
from .calib import apply_card, roi_px, overlaps


@dataclass
class PartResult:
    id: int
    bbox: tuple
    diameter_px: float
    diameter_mm: float | None
    roundness: float
    solidity: float
    hole_count: int
    hole_ratio: float | None      # hole diameter / outer diameter
    hole_offset: float | None     # hole centre offset, as a fraction of the outer radius
    rust_ratio: float
    scratch_ratio: float
    mean_L: float
    passed: bool
    reasons: list = field(default_factory=list)


def _centroid(c):
    m = cv2.moments(c)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(c)
        return x + w / 2, y + h / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def inspect(img_bgr: np.ndarray, cfg: dict) -> list:
    cfg = cfg["parts"] if "parts" in cfg else cfg
    cal = cfg.get("calibration")
    if cal:
        img_bgr = apply_card(img_bgr, cal["card_roi"], cal.get("target_gray", 128))
    mask = foreground_mask(img_bgr, cfg)
    h, w = mask.shape
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    card = roi_px(img_bgr, cal["card_roi"]) if cal else None
    min_area = cfg["seg"]["min_area_frac"] * h * w
    outer = [i for i in range(len(cnts)) if hier[i][3] == -1 and cv2.contourArea(cnts[i]) >= min_area
             and not (card and overlaps(cv2.boundingRect(cnts[i]), card))]

    def order(i):
        x, y, bw, bh = cv2.boundingRect(cnts[i])
        return (y + bh / 2) // 80, x + bw / 2
    outer.sort(key=order)

    L = _lab_float(img_bgr)[..., 0]
    gray8 = np.clip(L * 255.0 / 100.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sigma = cfg["surface"].get("denoise_sigma", 0.8)
    gray_f = cv2.GaussianBlur(gray8, (0, 0), sigma) if sigma else gray8     # light denoise before scratch search

    geo = []
    for i in outer:
        c = cnts[i]
        area = cv2.contourArea(c)
        d = math.sqrt(4 * area / math.pi)
        holes = [j for j in range(len(cnts)) if hier[j][3] == i
                 and cv2.contourArea(cnts[j]) >= cfg["hole"]["min_area_frac"] * area]
        geo.append({"i": i, "c": c, "area": area, "d": d, "holes": holes})
    med_d = float(np.median([g["d"] for g in geo])) if geo else 0.0
    ratios = []
    for g in geo:
        if g["holes"]:
            hc = max((cnts[j] for j in g["holes"]), key=cv2.contourArea)
            g["hc"] = hc
            g["hole_d"] = math.sqrt(4 * cv2.contourArea(hc) / math.pi)
            ratios.append(g["hole_d"] / g["d"])
    med_ratio = float(np.median(ratios)) if ratios else None

    px_per_mm = cfg.get("px_per_mm")
    sz, hl, sh, sf = cfg["size"], cfg["hole"], cfg.get("shape") or {}, cfg["surface"]
    results = []
    for n, g in enumerate(geo, start=1):
        c, d = g["c"], g["d"]
        x, y, bw, bh = cv2.boundingRect(c)
        if len(c) >= 5:
            (_, _), (a1, a2), _ = cv2.fitEllipse(c)
            roundness = min(a1, a2) / (max(a1, a2) + 1e-9)
        else:
            roundness = 1.0
        solidity = g["area"] / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)

        hole_ratio = hole_offset = None
        if g["holes"]:
            hole_ratio = g["hole_d"] / d
            ox, oy = _centroid(c)
            hx, hy = _centroid(g["hc"])
            hole_offset = math.hypot(hx - ox, hy - oy) / (d / 2)

        # surface region: the ring between the hole and the rim, eroded to skip edges
        local = np.zeros((bh, bw), np.uint8)
        shift = np.array([x, y])
        cv2.drawContours(local, [c - shift], -1, 255, -1)
        for j in g["holes"]:
            cv2.drawContours(local, [cnts[j] - shift], -1, 0, -1)
        er = max(3, int(0.12 * d)) | 1
        ring = cv2.erode(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er))) > 0
        if ring.sum() < 20:
            ring = local > 0
        Lc = L[y:y + bh, x:x + bw]
        hsvc = hsv[y:y + bh, x:x + bw]
        lo, hi = sf["rust_hue"]
        rust = (hsvc[..., 0] >= lo) & (hsvc[..., 0] <= hi) & (hsvc[..., 1] >= sf["rust_min_sat"]) & (hsvc[..., 2] >= 40)
        rust_in = rust & ring
        rust_ratio = float(rust_in.sum() / ring.sum())
        rust_zone = cv2.dilate(rust_in.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # not a scratch
        k = max(5, int(0.12 * d)) | 1
        crop = gray_f[y:y + bh, x:x + bw]
        bh_img = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        # noise-aware threshold: estimate sensor noise from the surface itself (robust MAD), so that a noisy
        # image raises the bar instead of producing false scratches. Subtle scratches are then not detectable.
        resid = (crop.astype(np.float32) - cv2.GaussianBlur(crop, (0, 0), 1.0))[ring]
        noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        thr = max(sf["bh_thresh"], sf.get("noise_k", 4.5) * noise)
        scratch_ratio = float(((bh_img > thr) & ring & ~rust_zone).sum() / ring.sum())

        reasons = []
        if px_per_mm and sz.get("min_mm") and sz.get("max_mm"):
            d_mm = d / px_per_mm
            if d_mm < sz["min_mm"]:
                reasons.append("SIZE_SMALL")
            elif d_mm > sz["max_mm"]:
                reasons.append("SIZE_LARGE")
        else:
            d_mm = None
            if d < med_d * (1 - sz["rel_tol"]):
                reasons.append("SIZE_SMALL")
            elif d > med_d * (1 + sz["rel_tol"]):
                reasons.append("SIZE_LARGE")
        if hole_ratio is None:
            if med_ratio is not None:
                reasons.append("HOLE_MISSING")
        else:
            if med_ratio and abs(hole_ratio - med_ratio) / med_ratio > hl["rel_tol"]:
                reasons.append("HOLE_SIZE")
            if hole_offset > hl["max_offset"]:
                reasons.append("HOLE_OFFCENTER")
        if (sh.get("min_roundness") and roundness < sh["min_roundness"]) or \
           (sh.get("min_solidity") and solidity < sh["min_solidity"]):
            reasons.append("DEFORMED")
        if rust_ratio > sf["max_rust_ratio"]:
            reasons.append("RUST")
        if scratch_ratio > sf["max_scratch_ratio"]:
            reasons.append("SCRATCH")

        results.append(PartResult(n, (x, y, bw, bh), d, d_mm, roundness, solidity, len(g["holes"]),
                                  hole_ratio, hole_offset, rust_ratio, scratch_ratio,
                                  float(Lc[ring].mean()), not reasons, reasons))
    return results
