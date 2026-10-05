"""SYNTHETIC metal washers with injectable defects, so the pipeline can be tested without real parts.

These are drawings, NOT photographs of real metal. Real parts add glare, oil, burrs and texture that
this generator does not model. `severity` (0-1) scales how obvious each defect is, which lets us
measure where detection starts to fail.

  python tools/make_parts_synthetic.py        # writes samples/parts_*.png and samples/golden_parts/
"""
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

W, H = 640, 480
BG = (58, 60, 64)
BASE = np.array((150, 156, 162), np.float32)
RUST = np.array((40, 90, 150), np.float32)
POS = [(110, 130), (240, 130), (370, 130), (500, 130), (110, 300), (240, 300), (370, 300), (500, 300)]
KINDS = ["good", "chip", "nohole", "big", "small", "holesmall", "offcenter", "rust", "scratch"]
EXPECT = {"chip": "DEFORMED", "nohole": "HOLE_MISSING", "big": "SIZE_LARGE", "small": "SIZE_SMALL",
          "holesmall": "HOLE_SIZE", "offcenter": "HOLE_OFFCENTER", "rust": "RUST", "scratch": "SCRATCH"}
GROUP = {"chip": "Shape", "big": "Shape", "small": "Shape", "nohole": "Hole", "holesmall": "Hole",
         "offcenter": "Hole", "rust": "Surface", "scratch": "Surface"}


def draw_card(img, roi):
    x, y, rw, rh = int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)
    img[y:y + rh, x:x + rw] = 128
    return img


def draw_washer(img, c, kind, rng, sev=1.0):
    cx, cy = float(c[0]), float(c[1])
    R, r = 40.0 * rng.uniform(0.98, 1.02), 17.0 * rng.uniform(0.98, 1.02)
    hox = hoy = 0.0
    if kind == "big":
        R *= 1 + 0.16 * sev
    elif kind == "small":
        R *= 1 - 0.16 * sev
    elif kind == "holesmall":
        r *= 1 - 0.40 * sev
    elif kind == "nohole":
        r = 0.0                      # a missing hole is binary, severity does not apply
    elif kind == "offcenter":
        a = rng.uniform(0, 2 * np.pi)
        hox, hoy = 0.22 * R * sev * np.cos(a), 0.22 * R * sev * np.sin(a)
    half = int(R * 1.25) + 4
    x0, y0 = int(cx) - half, int(cy) - half
    yy, xx = np.mgrid[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy)
    alpha = np.clip(R - d + 0.5, 0, 1)
    if r > 0.5:
        alpha *= np.clip(np.hypot(xx - (cx + hox), yy - (cy + hoy)) - r + 0.5, 0, 1)
    if kind == "chip":
        a = rng.uniform(0, 2 * np.pi)
        bx, by, b = cx + R * np.cos(a), cy + R * np.sin(a), 0.38 * R * sev
        alpha *= np.clip(np.hypot(xx - bx, yy - by) - b + 0.5, 0, 1)
    ang = np.arctan2(yy - cy, xx - cx)
    shade = 0.92 + 0.12 * (1 - np.clip(d / R, 0, 1)) + 0.04 * np.sin(10 * ang)
    shade *= 1 - 0.15 * np.clip((d - 0.88 * R) / (0.12 * R), 0, 1)
    col = BASE[None, None] * shade[..., None] + rng.normal(0, 3, (2 * half, 2 * half, 3))
    if kind == "rust":
        for _ in range(1 + int(2 * sev)):
            a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(r * 1.4 + 3, R * 0.75)
            px, py = cx + rr * np.cos(a), cy + rr * np.sin(a)
            br = 0.15 * R * (0.15 + 0.85 * sev)
            blob = (np.clip(br - np.hypot(xx - px, yy - py) + 0.5, 0, 1) * (0.25 + 0.5 * sev))[..., None]
            col = col * (1 - blob) + RUST[None, None] * blob
    if kind == "scratch":
        m = np.zeros((2 * half, 2 * half), np.uint8)
        for _ in range(1 + int(2 * sev)):
            a1 = rng.uniform(0, 2 * np.pi)
            a2 = a1 + np.pi + rng.uniform(-0.7, 0.7)
            p1 = (int(half + 0.85 * R * np.cos(a1)), int(half + 0.85 * R * np.sin(a1)))
            p2 = (int(half + 0.85 * R * np.cos(a2)), int(half + 0.85 * R * np.sin(a2)))
            cv2.line(m, p1, p2, 255, 1, cv2.LINE_AA)
        k = (m.astype(np.float32) / 255.0 * (0.05 + 0.55 * sev))[..., None]
        col = col * (1 - k)
    region = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    a3 = alpha[..., None]
    img[y0:y0 + 2 * half, x0:x0 + 2 * half] = region * (1 - a3) + col * a3


def washer_image(kinds, rng, sev=1.0, card_roi=(0.88, 0.86, 0.08, 0.08)):
    img = np.full((H, W, 3), BG, np.float32) + rng.normal(0, 2, (H, W, 3))
    for (cx, cy), k in zip(POS, kinds):
        draw_washer(img, (cx + rng.integers(-8, 9), cy + rng.integers(-8, 9)), k, rng, sev)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return draw_card(img, card_roi) if card_roi else img


def main():
    rng = np.random.default_rng(11)
    (ROOT / "samples/golden_parts").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "samples/parts_batch_good.png"), washer_image(["good"] * 8, rng))
    cv2.imwrite(str(ROOT / "samples/parts_batch_mixed.png"),
                washer_image(["good", "chip", "nohole", "big", "holesmall", "offcenter", "rust", "scratch"], rng))
    for i in range(8):
        cv2.imwrite(str(ROOT / f"samples/golden_parts/good_{i}.png"), washer_image(["good"] * 8, rng))
    print("wrote synthetic parts samples")


if __name__ == "__main__":
    main()
