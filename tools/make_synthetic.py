"""Generate SYNTHETIC demo images so the pipeline can be smoke-tested without any real photos.

These are cartoon-like drawings, NOT real food. They only prove the code runs end-to-end.
Replace them with real photos from your own jig before judging accuracy.

  python tools/make_synthetic.py            # writes samples/ and refs/
"""
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partguard.config import load_config          # noqa: E402
from partguard.bento import crop_roi              # noqa: E402

W, H = 640, 480

# ---------------------------------------------------------------- bakery
def draw_bun(img, c, axes, base, rng, spots=0):
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    r2 = ((xx - c[0]) / axes[0]) ** 2 + ((yy - c[1]) / axes[1]) ** 2
    shade = (1 - 0.15 * np.clip(r2, 0, 1))[..., None]
    bun = np.array(base, np.float32)[None, None] * shade + rng.normal(0, 4, (H, W, 3))
    for _ in range(spots):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.1, 0.5)
        sx, sy = int(c[0] + np.cos(a) * rr * axes[0]), int(c[1] + np.sin(a) * rr * axes[1])
        cv2.circle(bun, (sx, sy), 9, (18, 28, 38), -1)
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(c[0]), int(c[1])), (int(axes[0]), int(axes[1])), 0, 0, 360, 255, -1, cv2.LINE_AA)
    a = (cv2.GaussianBlur(m, (3, 3), 0) / 255.0)[..., None]
    return np.clip(img * (1 - a) + bun * a, 0, 255).astype(np.float32)


GOOD = (72, 138, 196)      # BGR golden brown
PALE = (150, 190, 225)
DARK = (28, 55, 85)

def draw_card(img, roi):
    x, y, rw, rh = int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)
    img[y:y + rh, x:x + rw] = 128
    return img


def bakery_image(kinds, rng, card_roi=(0.88, 0.86, 0.08, 0.08)):
    img = np.full((H, W, 3), (212, 214, 216), np.float32) + rng.normal(0, 2, (H, W, 3))
    pos = [(130, 120), (320, 120), (510, 120), (130, 330), (320, 330), (510, 330)]
    for (cx, cy), k in zip(pos, kinds):
        cx += rng.integers(-8, 9); cy += rng.integers(-8, 9)
        j = rng.uniform(0.95, 1.05)
        if k == "good":      img = draw_bun(img, (cx, cy), (50, 48), np.array(GOOD) * j, rng)
        elif k == "pale":    img = draw_bun(img, (cx, cy), (50, 48), PALE, rng)
        elif k == "dark":    img = draw_bun(img, (cx, cy), (50, 48), DARK, rng)
        elif k == "small":   img = draw_bun(img, (cx, cy), (32, 31), np.array(GOOD) * j, rng)
        elif k == "oval":    img = draw_bun(img, (cx, cy), (64, 34), np.array(GOOD) * j, rng)
        elif k == "burnt":   img = draw_bun(img, (cx, cy), (50, 48), np.array(GOOD) * j, rng, spots=5)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return draw_card(img, card_roi) if card_roi else img

# ---------------------------------------------------------------- bento
def paint_dish(w, h, kind, rng):
    base = {"empty": (150, 160, 170), "rice": (240, 242, 244), "fried_chicken": (60, 130, 190),
            "greens": (60, 150, 70), "egg": (70, 205, 240), "fish": (105, 130, 235)}[kind]
    jitter = rng.uniform(0.93, 1.07)
    img = np.clip(np.array(base, np.float32) * jitter + rng.normal(0, 6, (h, w, 3)), 0, 255)
    img = img.astype(np.uint8)
    extra = {"rice": [((225, 228, 230), 40)], "fried_chicken": [((30, 70, 120), 14)],
             "greens": [((30, 90, 40), 26), ((90, 190, 110), 14)],
             "egg": [((245, 245, 245), 12)], "fish": [((235, 240, 250), 10)], "empty": []}[kind]
    for col, n in extra:
        for _ in range(n):
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(rng.integers(6, 18)), col, -1)
    return img


def bento_image(cfg, assignment, rng, card_roi=(0.44, 0.95, 0.12, 0.04)):
    img = np.full((H, W, 3), (45, 45, 50), np.uint8)
    for name, (x, y, w, h) in cfg["bento"]["tray"]["compartments"].items():
        x0, y0, pw, ph = int(x * W), int(y * H), int(w * W), int(h * H)
        img[y0:y0 + ph, x0:x0 + pw] = paint_dish(pw, ph, assignment[name], rng)
    return draw_card(img, card_roi) if card_roi else img


def main():
    rng = np.random.default_rng(7)
    (ROOT / "samples").mkdir(exist_ok=True)

    cv2.imwrite(str(ROOT / "samples/bakery_batch_good.png"),
                bakery_image(["good"] * 6, rng))
    cv2.imwrite(str(ROOT / "samples/bakery_batch_mixed.png"),
                bakery_image(["good", "pale", "dark", "small", "oval", "burnt"], rng))
    for i in range(3):                                      # extra "golden" samples for threshold tuning
        (ROOT / "samples/golden").mkdir(exist_ok=True)
        cv2.imwrite(str(ROOT / f"samples/golden/good_{i}.png"), bakery_image(["good"] * 6, rng))

    cfg = load_config(ROOT / "configs/bento.yaml")
    comps = cfg["bento"]["tray"]["compartments"]
    inset = cfg["bento"]["tray"]["inset"]
    for label in ["empty", "rice", "fried_chicken", "greens", "egg", "fish"]:   # reference library
        d = ROOT / "refs" / label
        d.mkdir(parents=True, exist_ok=True)
        for k in range(6):
            name = list(comps)[k % 4]
            full = bento_image(cfg, {n: label for n in comps}, rng)
            crop, _ = crop_roi(full, comps[name], inset)
            cv2.imwrite(str(d / f"{k}.png"), crop)

    A = {"rice": "rice", "main": "fried_chicken", "side1": "greens", "side2": "egg"}
    cv2.imwrite(str(ROOT / "samples/bento_A_ok.png"), bento_image(cfg, A, rng))
    cv2.imwrite(str(ROOT / "samples/bento_A_missing_side.png"),
                bento_image(cfg, {**A, "side1": "empty"}, rng))
    cv2.imwrite(str(ROOT / "samples/bento_A_wrong_main.png"),
                bento_image(cfg, {**A, "main": "fish"}, rng))
    print("wrote synthetic samples/ and refs/")


if __name__ == "__main__":
    main()
