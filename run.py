"""PartGuard CLI.

Examples
  python run.py --mode parts  --source samples/parts_batch_mixed.png
  python run.py --mode bakery --source samples/bakery_batch_mixed.png
  python run.py --mode bento  --source samples/ --menu A_chicken
  python run.py --mode bento  --source 0 --menu A_chicken --show        # webcam, press q to quit
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2

from partguard.config import load_config
from partguard import bakery, parts
from partguard.backends import make_embedder
from partguard.bento import BentoVerifier, IMG_EXT
from partguard.report import (summarize_bakery, summarize_bento, draw_bakery, draw_bento,
                              JsonlLogger)

ROOT = Path(__file__).resolve().parent


def build(args):
    if args.mode == "parts":
        cfg = load_config(args.config or ROOT / "configs/parts.yaml")

        def check(img):
            res = parts.inspect(img, cfg)
            return (all(r.passed for r in res) and len(res) > 0, res,
                    draw_bakery(img, res), summarize_bakery(res, args.lang))
        return check

    if args.mode == "bakery":
        cfg = load_config(args.config or ROOT / "configs/bakery.yaml")

        def check(img):
            res = bakery.inspect(img, cfg)
            return (all(r.passed for r in res) and len(res) > 0, res,
                    draw_bakery(img, res), summarize_bakery(res, args.lang))
        return check

    cfg = load_config(args.config or ROOT / "configs/bento.yaml")
    emb = make_embedder(args.embedder, **({"model_path": args.model} if args.model else {}))
    ver = BentoVerifier(cfg, emb, args.refs)

    def check(img):
        res = ver.verify(img, args.menu)
        return res.passed, res, draw_bento(img, res), summarize_bento(res, args.lang)
    return check


def sources(src):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield "camera", frame
        cap.release()
        return
    p = Path(src)
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[skip] cannot read {f}")
            continue
        yield f, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["parts", "bakery", "bento"], required=True)
    ap.add_argument("--source", required=True, help="image file, folder, or camera index (0)")
    ap.add_argument("--config")
    ap.add_argument("--menu", default="A_chicken", help="bento: which menu to verify against")
    ap.add_argument("--refs", default=str(ROOT / "refs"), help="bento: reference image library")
    ap.add_argument("--embedder", default="hsv-hist", choices=["hsv-hist", "onnx", "hailo"])
    ap.add_argument("--model", help="path to ONNX model when --embedder onnx")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--out", default=str(ROOT / "out"), help="folder for annotated images")
    ap.add_argument("--log", default=str(ROOT / "logs/inspections.jsonl"))
    ap.add_argument("--show", action="store_true", help="open a preview window")
    args = ap.parse_args()

    check = build(args)
    logger = JsonlLogger(args.log)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lat, n_pass, n = [], 0, 0

    for name, img in sources(args.source):
        t0 = time.perf_counter()
        passed, res, annotated, lines = check(img)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms); n += 1; n_pass += int(passed)
        logger.write(args.mode, name, passed, ms, res)
        print(f"\n[{Path(str(name)).name}] {'PASS' if passed else 'FAIL'}  ({ms:.0f} ms)")
        print("\n".join(lines))
        stem = Path(str(name)).stem if name != "camera" else f"cam_{n:05d}"
        if name != "camera" or not passed:
            cv2.imwrite(str(out / f"{stem}_checked.png"), annotated)
        if args.show:
            cv2.imshow("PartGuard", annotated)
            if cv2.waitKey(1 if name == "camera" else 0) & 0xFF == ord("q"):
                break

    if n:
        s = sorted(lat)
        print(f"\n--- {n} inspected, {n_pass} pass | latency median {s[len(s)//2]:.0f} ms, "
              f"max {s[-1]:.0f} ms (CPU, laptop) ---")


if __name__ == "__main__":
    main()
