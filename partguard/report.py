"""Turn raw results into: a human-readable message, an annotated image, and a JSONL log line.

NOTE: OpenCV's putText cannot draw Chinese glyphs, so on-image labels use short ASCII codes.
The text summaries (console / log / UI) are available in Chinese and English.
"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import json
import cv2

TEXT = {
    "zh": {
        "PASS": "合格", "FAIL": "不合格",
        "SIZE_SMALL": "尺寸偏小", "SIZE_LARGE": "尺寸偏大", "DEFORMED": "外形變形",
        "UNDERBAKED": "烘烤不足（顏色偏淺）", "OVERBAKED": "烘烤過度（顏色偏深）",
        "BURNT_SPOTS": "有焦黑斑點", "UNEVEN_COLOR": "上色不均",
        "HOLE_MISSING": "缺少孔洞", "HOLE_SIZE": "孔徑異常", "HOLE_OFFCENTER": "孔位偏心",
        "RUST": "有銹斑", "SCRATCH": "有刮痕",
        "MISSING": "缺少內容物", "WRONG_ITEM": "品項錯誤", "UNRECOGNIZED": "無法辨識",
        "bakery_head": "共 {n} 個，{ok} 個合格、{bad} 個需處理",
        "bento_head": "餐點 {menu}：{state}",
        "item": "第 {id} 個", "comp": "格位「{name}」",
        "expect": "應為「{e}」，偵測到「{p}」",
    },
    "en": {
        "PASS": "PASS", "FAIL": "FAIL",
        "SIZE_SMALL": "too small", "SIZE_LARGE": "too large", "DEFORMED": "deformed shape",
        "UNDERBAKED": "under-baked (too pale)", "OVERBAKED": "over-baked (too dark)",
        "BURNT_SPOTS": "burnt spots", "UNEVEN_COLOR": "uneven colour",
        "HOLE_MISSING": "hole missing", "HOLE_SIZE": "hole size off", "HOLE_OFFCENTER": "hole off-centre",
        "RUST": "rust spots", "SCRATCH": "scratches",
        "MISSING": "missing item", "WRONG_ITEM": "wrong item", "UNRECOGNIZED": "unrecognized",
        "bakery_head": "{n} items: {ok} pass, {bad} need attention",
        "bento_head": "Menu {menu}: {state}",
        "item": "Item {id}", "comp": "Compartment '{name}'",
        "expect": "expected '{e}', detected '{p}'",
    },
}


def summarize_bakery(results, lang="zh"):
    t = TEXT[lang]
    bad = [r for r in results if not r.passed]
    lines = [t["bakery_head"].format(n=len(results), ok=len(results) - len(bad), bad=len(bad))]
    sep = "、" if lang == "zh" else ", "
    for r in bad:
        lines.append(f"- {t['item'].format(id=r.id)}: " + sep.join(t[c] for c in r.reasons))
    return lines


def summarize_bento(res, lang="zh"):
    t = TEXT[lang]
    lines = [t["bento_head"].format(menu=res.menu, state=t["PASS"] if res.passed else t["FAIL"])]
    for c in res.comps:
        if not c.ok:
            lines.append(f"- {t['comp'].format(name=c.name)}: {t[c.reason]}（"
                         + t["expect"].format(e=c.expected, p=c.predicted) + "）")
    return lines


GREEN, RED = (60, 180, 60), (40, 40, 220)


def draw_bakery(img, results):
    out = img.copy()
    for r in results:
        x, y, w, h = r.bbox
        col = GREEN if r.passed else RED
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label = f"#{r.id} " + ("OK" if r.passed else r.reasons[0])
        cv2.putText(out, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out


def draw_bento(img, res):
    out = img.copy()
    for c in res.comps:
        x, y, w, h = c.rect_px
        col = GREEN if c.ok else RED
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 3)
        label = f"{c.name}: " + ("OK" if c.ok else f"{c.reason} ({c.predicted})")
        cv2.putText(out, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(out, "PASS" if res.passed else "FAIL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, GREEN if res.passed else RED, 3)
    return out


def _plain(o):
    if is_dataclass(o):
        return {k: _plain(v) for k, v in asdict(o).items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if hasattr(o, "item"):          # numpy scalar
        return o.item()
    return o


class JsonlLogger:
    """Append one JSON line per inspection (also useful as Stage II performance evidence)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, mode, source, passed, latency_ms, details):
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "mode": mode,
               "source": str(source), "passed": bool(passed),
               "latency_ms": round(latency_ms, 1), "details": _plain(details)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
