"""Evaluate the bakery inspector on a PUBLIC good/bad bread image set that you download yourself.

Arrange the images as:   DATA/good/*.jpg   and   DATA/bad/*.jpg   (folder names are configurable).
Thresholds are FITTED on half of the good images, then everything is scored on held-out images,
so the numbers are not tuned on the test set.

  python tools/eval_public_bread.py --data path/to/bread --good good --bad bad

IMPORTANT: public 'bad' labels often mean stale/old bread, which may not be visible in colour, size or
shape. A low detection rate is a legitimate finding, so report whatever you get. Check the dataset's
licence and cite it. If you cannot get the dataset, tools/benchmark_synthetic.py still runs offline.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partguard import bakery                      # noqa: E402
from partguard.config import load_config          # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load(folder: Path, limit, max_side, rng):
    files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in EXT)
    rng.shuffle(files)
    out = []
    for f in files[:limit]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append((f.name, img))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--good", default="good")
    ap.add_argument("--bad", default="bad")
    ap.add_argument("--limit", type=int, default=400, help="max images per class")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    root = Path(a.data)
    good = load(root / a.good, a.limit, a.max_side, rng)
    bad = load(root / a.bad, a.limit, a.max_side, rng)
    if len(good) < 20 or len(bad) < 10:
        sys.exit(f"found {len(good)} good / {len(bad)} bad images: need at least 20 / 10")
    rng.shuffle(good)
    half = len(good) // 2
    fit, good_test = good[:half], good[half:]

    cfg = load_config(ROOT / "configs/bakery.yaml")
    cfg["bakery"].pop("calibration", None)        # public photos have no gray card

    # fit thresholds on good items of the fit split
    stats = []
    for _, img in fit:
        for r in bakery.inspect(img, cfg):
            stats.append((r.mean_L, r.std_L, r.roundness, r.solidity))
    if len(stats) < 10:
        sys.exit("segmentation found too few items in the fit split; check the image background")
    m, s = np.mean(stats, 0), np.std(stats, 0)
    fitted = copy.deepcopy(cfg)
    c = fitted["bakery"]
    c["color"]["L_min"], c["color"]["L_max"] = float(m[0] - 3 * s[0]), float(m[0] + 3 * s[0])
    c["color"]["max_std_L"] = float(m[1] + 3 * s[1])
    c["shape"]["min_roundness"] = float(max(0.3, m[2] - 3 * s[2]))
    c["shape"]["min_solidity"] = float(max(0.3, m[3] - 3 * s[3]))

    def flagged(img):
        res = bakery.inspect(img, fitted)
        return (not res) or any(not r.passed for r in res), len(res)

    g = [flagged(img) for _, img in good_test]
    b = [flagged(img) for _, img in bad]
    fp, tn = sum(x[0] for x in g), len(g)
    tp, pn = sum(x[0] for x in b), len(b)
    none_found = sum(1 for x in g + b if x[1] == 0)
    rlo, rhi = wilson(tp, pn)
    flo, fhi = wilson(fp, tn)
    res = {"n_fit_good": len(fit), "n_test_good": tn, "n_test_bad": pn,
           "detect_bad_pct": 100 * tp / pn, "detect_ci95": [100 * rlo, 100 * rhi],
           "false_alarm_good_pct": 100 * fp / tn, "false_alarm_ci95": [100 * flo, 100 * fhi],
           "images_with_no_item_found": none_found,
           "fitted_thresholds": {"L_min": c["color"]["L_min"], "L_max": c["color"]["L_max"],
                                 "max_std_L": c["color"]["max_std_L"],
                                 "min_roundness": c["shape"]["min_roundness"],
                                 "min_solidity": c["shape"]["min_solidity"]}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "public_bread_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
