"""Computer-only benchmark: how well do the checks hold up when conditions change?

Everything here is SYNTHETIC (cartoon-like images with injected faults). It measures the *method's
sensitivity* to lighting, colour cast, noise and tray misalignment, with and without the gray-card
calibration. It does NOT measure accuracy on real food.

  python tools/make_synthetic.py            # once, builds refs/
  python tools/benchmark_synthetic.py       # writes results/summary.json, CSVs and PNG charts
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import make_synthetic as ms                                   # noqa: E402
from partguard import bakery                                  # noqa: E402
from partguard.backends import make_embedder                  # noqa: E402
from partguard.bento import BentoVerifier                     # noqa: E402
from partguard.calib import roi_px                            # noqa: E402
from partguard.config import load_config                      # noqa: E402

W, H = ms.W, ms.H
DISHES = ["rice", "fried_chicken", "greens", "egg", "fish"]
BAKERY_NOMINAL = [(130, 120), (320, 120), (510, 120), (130, 330), (320, 330), (510, 330)]
EXPECTED_CODE = {"pale": "UNDERBAKED", "dark": "OVERBAKED", "small": "SIZE_SMALL",
                 "oval": "DEFORMED", "burnt": "BURNT_SPOTS"}

SWEEPS = {
    "gain": ("Lighting brightness (x)", [0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4], "gain"),
    "tilt": ("Colour cast (warm +, cool -)", [-0.2, -0.1, 0.0, 0.1, 0.2], "tilt"),
    "noise": ("Sensor noise (std, 0-255)", [0, 5, 10, 20, 30], "noise"),
    "shift": ("Tray misalignment (px)", [0, 5, 10, 20, 30, 40], "shift"),
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def remove_card(img, card_roi, fill):
    out = img.copy()
    x, y, w, h = roi_px(img, card_roi)
    out[y:y + h, x:x + w] = fill
    return out


def perturb(img, rng, card_roi=None, gain=1.0, tilt=0.0, noise=0.0, shift=0, exposure=1.0):
    out = img
    if shift:
        M = np.float32([[1, 0, shift], [0, 1, shift // 2]])
        moved = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(45, 45, 50))
        if card_roi is not None:                           # the card is fixed to the camera, not the tray
            x, y, w, h = roi_px(img, card_roi)
            moved[y:y + h, x:x + w] = img[y:y + h, x:x + w]
        out = moved
    f = out.astype(np.float32) * np.array([1 - tilt, 1.0, 1 + tilt], np.float32) * gain * exposure
    if noise:
        f += rng.normal(0, noise, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ scene sets
def make_bento_set(cfg, n, rng):
    menus = cfg["bento"]["menus"]
    card = cfg["bento"]["calibration"]["card_roi"]
    scenes = []
    for _ in range(n):
        menu = str(rng.choice(list(menus)))
        spec = dict(menus[menu])
        assign, err = dict(spec), None
        if rng.random() < 0.5:
            comp = str(rng.choice(list(spec)))
            kind = str(rng.choice(["MISSING", "WRONG_ITEM"]))
            if kind == "MISSING":
                assign[comp] = "empty"
            else:
                assign[comp] = str(rng.choice([d for d in DISHES if d != spec[comp]]))
            err = (comp, kind)
        scenes.append({"img": ms.bento_image(cfg, assign, rng, card), "menu": menu, "err": err})
    return scenes


def make_bakery_set(cfg, n, rng):
    card = cfg["bakery"]["calibration"]["card_roi"]
    kinds_all = ["good", "pale", "dark", "small", "oval", "burnt"]
    p = [0.70] + [0.06] * 5
    scenes = []
    for _ in range(n):
        kinds = [str(k) for k in rng.choice(kinds_all, size=6, p=p)]
        scenes.append({"img": ms.bakery_image(kinds, rng, card), "kinds": kinds})
    return scenes


# ------------------------------------------------------------------ evaluation
def eval_bento(ver, scenes, cond, card_roi, seed, use_card=True, exposure=1.0):
    rng = np.random.default_rng(seed)
    tp = fp = pos = neg = exact = 0
    lat = []
    for sc in scenes:
        base = sc["img"] if use_card else remove_card(sc["img"], card_roi, (45, 45, 50))
        img = perturb(base, rng, card_roi if use_card else None, exposure=exposure, **cond)
        t0 = time.perf_counter()
        res = ver.verify(img, sc["menu"])
        lat.append((time.perf_counter() - t0) * 1000)
        flagged = not res.passed
        if sc["err"]:
            pos += 1
            tp += flagged
            bad = {(c.name, c.reason) for c in res.comps if not c.ok}
            exact += int(flagged and bad == {sc["err"]})
        else:
            neg += 1
            fp += flagged
    return {"pos": pos, "tp": tp, "neg": neg, "fp": fp, "exact": exact, "lat": lat}


def eval_bakery(cfg, scenes, cond, card_roi, seed, use_card=True, exposure=1.0):
    rng = np.random.default_rng(seed)
    d_tot = d_hit = d_reason = g_tot = g_fp = seg_fail = 0
    lat = []
    for sc in scenes:
        base = sc["img"] if use_card else remove_card(sc["img"], card_roi, (212, 214, 216))
        img = perturb(base, rng, card_roi if use_card else None, exposure=exposure, **cond)
        t0 = time.perf_counter()
        res = bakery.inspect(img, cfg)
        lat.append((time.perf_counter() - t0) * 1000)
        # match each detected item to the nearest nominal slot
        slots = {}
        for r in res:
            cx, cy = r.bbox[0] + r.bbox[2] / 2, r.bbox[1] + r.bbox[3] / 2
            k = int(np.argmin([(cx - sx) ** 2 + (cy - sy) ** 2 for sx, sy in BAKERY_NOMINAL]))
            slots.setdefault(k, []).append(r)
        if len(res) != 6 or any(len(v) != 1 for v in slots.values()) or len(slots) != 6:
            seg_fail += 1
            continue
        for k, kind in enumerate(sc["kinds"]):
            r = slots[k][0]
            if kind == "good":
                g_tot += 1
                g_fp += int(not r.passed)
            else:
                d_tot += 1
                d_hit += int(not r.passed)
                d_reason += int(EXPECTED_CODE[kind] in r.reasons)
    return {"d_tot": d_tot, "d_hit": d_hit, "d_reason": d_reason, "g_tot": g_tot, "g_fp": g_fp,
            "seg_fail": seg_fail, "batches": len(scenes), "lat": lat}


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bento-n", type=int, default=200)
    ap.add_argument("--bakery-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--headroom", type=float, default=0.75, help="exposure factor for the headroom series")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bento_cfg = load_config(ROOT / "configs/bento.yaml")
    bakery_cfg = load_config(ROOT / "configs/bakery.yaml")
    bento_off = copy.deepcopy(bento_cfg)
    del bento_off["bento"]["calibration"]
    bakery_off = copy.deepcopy(bakery_cfg)
    del bakery_off["bakery"]["calibration"]
    emb = make_embedder("hsv-hist")
    ver_on = BentoVerifier(bento_cfg, emb, ROOT / "refs")
    ver_off = BentoVerifier(bento_off, emb, ROOT / "refs")
    bento_card = bento_cfg["bento"]["calibration"]["card_roi"]
    bakery_card = bakery_cfg["bakery"]["calibration"]["card_roi"]

    rng = np.random.default_rng(a.seed)
    bento_set = make_bento_set(bento_cfg, a.bento_n, rng)
    bakery_set = make_bakery_set(bakery_cfg, a.bakery_n, rng)
    print(f"scenes ready: {len(bento_set)} trays, {len(bakery_set)} batches (6 items each)")

    rows, all_lat = [], {"bento": [], "bakery": []}
    for sweep, (label, values, key) in SWEEPS.items():
        for v in values:
            cond = {key: v}
            for calib in ("off", "on", "on_headroom"):
                seed = a.seed + 1
                exp = a.headroom if calib == "on_headroom" else 1.0
                use = calib != "off"
                r = eval_bento(ver_off if calib == "off" else ver_on, bento_set, cond, bento_card, seed, use, exp)
                all_lat["bento"] += r["lat"]
                lo, hi = wilson(r["tp"], r["pos"])
                flo, fhi = wilson(r["fp"], r["neg"])
                rows.append({"mode": "bento", "sweep": sweep, "value": v, "calibration": calib,
                             "detect_pct": pct(r["tp"], r["pos"]), "detect_lo": 100 * lo, "detect_hi": 100 * hi,
                             "false_alarm_pct": pct(r["fp"], r["neg"]), "fa_lo": 100 * flo, "fa_hi": 100 * fhi,
                             "reason_ok_pct": pct(r["exact"], r["pos"]), "n_pos": r["pos"], "n_neg": r["neg"],
                             "seg_fail_pct": 0.0})
                if sweep != "shift":
                    seed = a.seed + 2
                    r = eval_bakery(bakery_off if calib == "off" else bakery_cfg, bakery_set, cond, bakery_card, seed, use, exp)
                    all_lat["bakery"] += r["lat"]
                    lo, hi = wilson(r["d_hit"], r["d_tot"])
                    flo, fhi = wilson(r["g_fp"], r["g_tot"])
                    rows.append({"mode": "bakery", "sweep": sweep, "value": v, "calibration": calib,
                                 "detect_pct": pct(r["d_hit"], r["d_tot"]), "detect_lo": 100 * lo, "detect_hi": 100 * hi,
                                 "false_alarm_pct": pct(r["g_fp"], r["g_tot"]), "fa_lo": 100 * flo, "fa_hi": 100 * fhi,
                                 "reason_ok_pct": pct(r["d_reason"], r["d_tot"]), "n_pos": r["d_tot"], "n_neg": r["g_tot"],
                                 "seg_fail_pct": pct(r["seg_fail"], r["batches"])})
            print(f"  done {sweep}={v}")

    with open(out / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    def lat_stats(x):
        s = sorted(x)
        return {"median_ms": round(s[len(s) // 2], 1), "p95_ms": round(s[int(0.95 * len(s))], 1), "n": len(s)}

    base = {m: next(r for r in rows if r["mode"] == m and r["sweep"] == "gain" and r["value"] == 1.0
                    and r["calibration"] == "on") for m in ("bento", "bakery")}
    summary = {"seed": a.seed, "synthetic": True, "latency": {k: lat_stats(v) for k, v in all_lat.items()},
               "baseline_on": base, "n_bento": a.bento_n, "n_bakery_batches": a.bakery_n}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for mode in ("bento", "bakery"):
            sw = [s for s in SWEEPS if s != "shift"] if mode == "bakery" else list(SWEEPS)
            fig, axes = plt.subplots(1, len(sw), figsize=(4 * len(sw), 3.2), sharey=True)
            for ax, s in zip(axes, sw):
                for calib, col in (("off", "#D6453D"), ("on", "#E08A2E"), ("on_headroom", "#2E9E6B")):
                    pts = [r for r in rows if r["mode"] == mode and r["sweep"] == s and r["calibration"] == calib]
                    ax.plot([p["value"] for p in pts], [p["detect_pct"] for p in pts], "-o", color=col,
                            label=f"detect, card {calib}")
                    ax.plot([p["value"] for p in pts], [p["false_alarm_pct"] for p in pts], "--s", color=col,
                            alpha=0.6, label=f"false alarm, card {calib}")
                ax.set_xlabel(SWEEPS[s][0]); ax.set_ylim(-3, 103); ax.grid(alpha=0.3)
            axes[0].set_ylabel("%"); axes[0].legend(fontsize=7)
            fig.suptitle(f"{mode}: synthetic robustness sweep", fontsize=10)
            fig.tight_layout(); fig.savefig(out / f"robustness_{mode}.png", dpi=130); plt.close(fig)
    except ImportError:
        print("matplotlib not installed: skipped PNG charts")

    print("\nLatency:", summary["latency"])
    print("Baseline (gain=1.0, card on):")
    for m, r in base.items():
        print(f"  {m}: detect {r['detect_pct']:.1f}%  false alarm {r['false_alarm_pct']:.1f}%")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
