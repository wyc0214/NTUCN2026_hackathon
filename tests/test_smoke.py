"""Smoke test on synthetic images. Run: python -m pytest -q   (or: python tests/test_smoke.py)"""
import subprocess
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partguard.config import load_config
from partguard import bakery, parts
from partguard.backends import make_embedder
from partguard.bento import BentoVerifier


def setup_module(_=None):
    subprocess.run([sys.executable, str(ROOT / "tools/make_synthetic.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/make_parts_synthetic.py")], check=True)


def test_bakery_good_batch_passes():
    cfg = load_config(ROOT / "configs/bakery.yaml")
    res = bakery.inspect(cv2.imread(str(ROOT / "samples/bakery_batch_good.png")), cfg)
    assert len(res) == 6 and all(r.passed for r in res)


def test_bakery_mixed_batch_flags_each_defect():
    cfg = load_config(ROOT / "configs/bakery.yaml")
    res = bakery.inspect(cv2.imread(str(ROOT / "samples/bakery_batch_mixed.png")), cfg)
    codes = [set(r.reasons) for r in res]
    assert codes[0] == set()
    assert "UNDERBAKED" in codes[1] and "OVERBAKED" in codes[2]
    assert "SIZE_SMALL" in codes[3] and "DEFORMED" in codes[4] and "BURNT_SPOTS" in codes[5]


def test_bento_detects_missing_and_wrong():
    ver = BentoVerifier(load_config(ROOT / "configs/bento.yaml"), make_embedder(), ROOT / "refs")
    ok = ver.verify(cv2.imread(str(ROOT / "samples/bento_A_ok.png")), "A_chicken")
    miss = ver.verify(cv2.imread(str(ROOT / "samples/bento_A_missing_side.png")), "A_chicken")
    wrong = ver.verify(cv2.imread(str(ROOT / "samples/bento_A_wrong_main.png")), "A_chicken")
    assert ok.passed
    assert not miss.passed and [c.reason for c in miss.comps if not c.ok] == ["MISSING"]
    assert not wrong.passed and [c.reason for c in wrong.comps if not c.ok] == ["WRONG_ITEM"]


def test_gray_card_calibration_restores_dim_light():
    import numpy as np
    cfg = load_config(ROOT / "configs/bakery.yaml")
    img = cv2.imread(str(ROOT / "samples/bakery_batch_good.png"))
    dim = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    assert all(r.passed for r in bakery.inspect(dim, cfg)) and len(bakery.inspect(dim, cfg)) == 6


def test_parts_good_batch_passes():
    cfg = load_config(ROOT / "configs/parts.yaml")
    res = parts.inspect(cv2.imread(str(ROOT / "samples/parts_batch_good.png")), cfg)
    assert len(res) == 8 and all(r.passed for r in res)


def test_parts_mixed_batch_flags_each_defect():
    cfg = load_config(ROOT / "configs/parts.yaml")
    res = parts.inspect(cv2.imread(str(ROOT / "samples/parts_batch_mixed.png")), cfg)
    want = [None, "DEFORMED", "HOLE_MISSING", "SIZE_LARGE", "HOLE_SIZE", "HOLE_OFFCENTER", "RUST", "SCRATCH"]
    assert len(res) == 8
    for r, w in zip(res, want):
        assert (w is None and r.passed) or (w in r.reasons), (r.id, r.reasons, w)


def test_parts_dim_light_is_corrected_by_gray_card():
    import numpy as np
    cfg = load_config(ROOT / "configs/parts.yaml")
    img = cv2.imread(str(ROOT / "samples/parts_batch_good.png"))
    dim = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    res = parts.inspect(dim, cfg)
    assert len(res) == 8 and all(r.passed for r in res)


if __name__ == "__main__":
    setup_module()
    test_bakery_good_batch_passes(); test_bakery_mixed_batch_flags_each_defect()
    test_bento_detects_missing_and_wrong(); test_gray_card_calibration_restores_dim_light()
    test_parts_good_batch_passes(); test_parts_mixed_batch_flags_each_defect(); test_parts_dim_light_is_corrected_by_gray_card()
    print("all smoke tests passed")
