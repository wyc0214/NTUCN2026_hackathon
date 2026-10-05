"""Computer-only benchmark for the metal-part inspector.

SYNTHETIC washers with injected defects (see make_parts_synthetic.py). It measures
  1. sensitivity to lighting, colour cast and sensor noise, with and without the gray-card calibration;
  2. the detection limit: how detection falls as defects get subtler (severity 0.25 .. 1.0).
It does NOT measure accuracy on real metal parts (no glare, oil, burrs or machining texture here).

  python tools/make_parts_synthetic.py && python tools/benchmark_parts.py
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import make_parts_synthetic as mp                 # noqa: E402
import benchmark_synthetic as bs                  # noqa: E402  (perturb, remove_card, wilson)
from partguard import parts                       # noqa: E402
from partguard.config import load_config          # noqa: E402

KINDS_P = ["good"] + mp.KINDS[1:]
P = [0.60] + [0.05] * 8
SWEEPS = {"gain": [0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4], "tilt": [-0.2, -0.1, 0.0, 0.1, 0.2],
          "noise": [0, 5, 10, 20, 30]}
SEVERITIES = [0.1, 0.25, 0.5, 0.75, 1.0]
NOMINAL_SEV = 0.75


def make_set(n, sev, card_roi, rng):
    out = []
    for _ in range(n):
        kinds = [str(k) for k in rng.choice(KINDS_P, size=8, p=P)]
        out.append({"img": mp.washer_image(kinds, rng, sev, card_roi), "kinds": kinds})
    return out


def evaluate(cfg, scenes, cond, card_roi, seed, use_card=True, exposure=1.0):
    rng = np.random.default_rng(seed)
    n_def = {k: 0 for k in mp.EXPECT}
    hit = {k: 0 for k in mp.EXPECT}
    reason = {k: 0 for k in mp.EXPECT}
    g_tot = g_fp = seg_fail = 0
    lat = []
    for sc in scenes:
        base = sc["img"] if use_card else bs.remove_card(sc["img"], card_roi, mp.BG)
        img = bs.perturb(base, rng, card_roi if use_card else None, exposure=exposure, **cond)
        t0 = time.perf_counter()
        res = parts.inspect(img, cfg)
        lat.append((time.perf_counter() - t0) * 1000)
        slots = {}
        for r in res:
            cx, cy = r.bbox[0] + r.bbox[2] / 2, r.bbox[1] + r.bbox[3] / 2
            k = int(np.argmin([(cx - sx) ** 2 + (cy - sy) ** 2 for sx, sy in mp.POS]))
            slots.setdefault(k, []).append(r)
        if len(res) != 8 or len(slots) != 8 or any(len(v) != 1 for v in slots.values()):
            seg_fail += 1
            continue
        for k, kind in enumerate(sc["kinds"]):
            r = slots[k][0]
            if kind == "good":
                g_tot += 1
                g_fp += int(not r.passed)
            else:
                n_def[kind] += 1
                hit[kind] += int(not r.passed)
                reason[kind] += int(mp.EXPECT[kind] in r.reasons)
    return {"n_def": n_def, "hit": hit, "reason": reason, "g_tot": g_tot, "g_fp": g_fp,
            "seg_fail": seg_fail, "batches": len(scenes), "lat": lat}


def agg(r, kinds=None):
    ks = kinds or list(mp.EXPECT)
    return sum(r["hit"][k] for k in ks), sum(r["n_def"][k] for k in ks), sum(r["reason"][k] for k in ks)


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--headroom", type=float, default=0.75)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg_on = load_config(ROOT / "configs/parts.yaml")
    cfg_off = copy.deepcopy(cfg_on)
    del cfg_off["parts"]["calibration"]
    card = cfg_on["parts"]["calibration"]["card_roi"]
    rng = np.random.default_rng(a.seed)
    nominal = make_set(a.n, NOMINAL_SEV, card, rng)
    print("nominal scenes ready", flush=True)

    rows, lat_all = [], []
    for sweep, values in SWEEPS.items():
        for v in values:
            cond = {sweep: v}
            for calib in ("off", "on", "on_headroom"):
                r = evaluate(cfg_off if calib == "off" else cfg_on, nominal, cond, card, a.seed + 1,
                             calib != "off", a.headroom if calib == "on_headroom" else 1.0)
                lat_all += r["lat"]
                hk, hn, hr = agg(r)
                lo, hi = bs.wilson(hk, hn)
                flo, fhi = bs.wilson(r["g_fp"], r["g_tot"])
                rows.append({"sweep": sweep, "value": v, "calibration": calib,
                             "detect_pct": pct(hk, hn), "detect_lo": 100 * lo, "detect_hi": 100 * hi,
                             "false_alarm_pct": pct(r["g_fp"], r["g_tot"]), "fa_lo": 100 * flo, "fa_hi": 100 * fhi,
                             "reason_ok_pct": pct(hr, hn), "n_def": hn, "n_good": r["g_tot"],
                             "seg_fail_pct": pct(r["seg_fail"], r["batches"])})
        print("  done", sweep, flush=True)
    with open(out / "benchmark_parts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items()} for r in rows)

    # detection limit vs severity (nominal light, calibration on)
    sev_rows, per_kind = [], {}
    for sev in SEVERITIES:
        scenes = nominal if sev == NOMINAL_SEV else make_set(a.n, sev, card, np.random.default_rng(a.seed + int(sev * 100)))
        r = evaluate(cfg_on, scenes, {}, card, a.seed + 3)
        lat_all += r["lat"]
        for grp in ("Shape", "Hole", "Surface"):
            ks = [k for k in mp.EXPECT if mp.GROUP[k] == grp]
            hk, hn, hr = agg(r, ks)
            lo, hi = bs.wilson(hk, hn)
            sev_rows.append({"severity": sev, "group": grp, "detect_pct": pct(hk, hn), "lo": 100 * lo,
                             "hi": 100 * hi, "n": hn, "reason_ok_pct": pct(hr, hn)})
        for k in mp.EXPECT:
            per_kind.setdefault(k, {})[str(sev)] = pct(r["hit"][k], r["n_def"][k])
        sev_rows.append({"severity": sev, "group": "Good parts flagged", "detect_pct": pct(r["g_fp"], r["g_tot"]),
                         "lo": 100 * bs.wilson(r["g_fp"], r["g_tot"])[0], "hi": 100 * bs.wilson(r["g_fp"], r["g_tot"])[1],
                         "n": r["g_tot"], "reason_ok_pct": float("nan")})
        print("  severity", sev, flush=True)
    with open(out / "severity_parts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sev_rows[0]))
        w.writeheader()
        w.writerows({k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items()} for r in sev_rows)

    s = sorted(lat_all)
    base = next(r for r in rows if r["sweep"] == "gain" and r["value"] == 1.0 and r["calibration"] == "on")
    summary = {"seed": a.seed, "synthetic": True, "n_batches": a.n, "parts_per_batch": 8,
               "nominal_severity": NOMINAL_SEV, "latency": {"median_ms": round(s[len(s) // 2], 1),
               "p95_ms": round(s[int(0.95 * len(s))], 1), "n": len(s)}, "baseline_on": base, "per_kind_by_severity": per_kind}
    (out / "summary_parts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    chart = {"labels": [("%g" % v) for v in SWEEPS["gain"]], "gain_fa": [], "sev_labels": [str(x) for x in SEVERITIES], "sev": {}}
    for calib in ("off", "on", "on_headroom"):
        pts = sorted([r for r in rows if r["sweep"] == "gain" and r["calibration"] == calib], key=lambda r: r["value"])
        chart["gain_fa"].append([round(r["false_alarm_pct"], 1) for r in pts])
    for grp in ("Shape", "Hole", "Surface", "Good parts flagged"):
        chart["sev"][grp] = [round(r["reason_ok_pct"] if grp != "Good parts flagged" else r["detect_pct"], 1) for sev in SEVERITIES for r in sev_rows if r["severity"] == sev and r["group"] == grp]
    (out / "chart_data_parts.json").write_text(json.dumps(chart), encoding="utf-8")
    print("latency", summary["latency"], "\nbaseline", {k: base[k] for k in ("detect_pct", "false_alarm_pct", "n_def", "n_good")}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
