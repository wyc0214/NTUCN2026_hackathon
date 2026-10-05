"""Bento tray check: crop each compartment, identify what is in it, compare with the menu."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from .calib import apply_card

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class CompResult:
    name: str
    expected: str
    predicted: str
    distance: float
    ok: bool
    reason: str | None        # MISSING / WRONG_ITEM / UNRECOGNIZED / None
    rect_px: tuple


@dataclass
class BentoResult:
    menu: str
    passed: bool
    comps: list


def crop_roi(img, rect_norm, inset=0.08):
    h, w = img.shape[:2]
    x, y, rw, rh = rect_norm
    x0, y0 = int((x + rw * inset) * w), int((y + rh * inset) * h)
    x1, y1 = int((x + rw * (1 - inset)) * w), int((y + rh * (1 - inset)) * h)
    return img[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


class BentoVerifier:
    def __init__(self, cfg: dict, embedder, refs_dir: str | Path):
        c = cfg["bento"] if "bento" in cfg else cfg
        self.comps = c["tray"]["compartments"]
        self.menus = c["menus"]
        self.inset = c["tray"].get("inset", 0.08)
        self.unknown_thresh = c["match"]["unknown_distance"]
        self.empty_label = c["match"].get("empty_label", "empty")
        self.calibration = c.get("calibration")
        self.embedder = embedder
        self.bank = self._load_refs(Path(refs_dir))   # list of (label, vector)
        if not self.bank:
            raise RuntimeError(f"no reference images found in {refs_dir}")

    def _load_refs(self, d: Path):
        bank = []
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    img = cv2.imread(str(f))
                    if img is not None:
                        bank.append((sub.name, self.embedder.embed(img)))
        return bank

    def classify(self, crop):
        v = self.embedder.embed(crop)
        best = {}
        for label, ref in self.bank:
            dist = float(np.linalg.norm(v - ref))
            if label not in best or dist < best[label]:
                best[label] = dist
        label, dist = min(best.items(), key=lambda kv: kv[1])
        if dist > self.unknown_thresh:
            return "unknown", dist
        return label, dist

    def verify(self, img, menu: str) -> BentoResult:
        spec = self.menus[menu]
        if self.calibration:
            img = apply_card(img, self.calibration["card_roi"], self.calibration.get("target_gray", 128))
        comps = []
        for name, rect in self.comps.items():
            crop, rect_px = crop_roi(img, rect, self.inset)
            pred, dist = self.classify(crop)
            exp = spec[name]
            ok = pred == exp
            reason = None
            if not ok:
                reason = ("MISSING" if pred == self.empty_label else
                          "UNRECOGNIZED" if pred == "unknown" else "WRONG_ITEM")
            comps.append(CompResult(name, exp, pred, dist, ok, reason, rect_px))
        return BentoResult(menu, all(c.ok for c in comps), comps)
