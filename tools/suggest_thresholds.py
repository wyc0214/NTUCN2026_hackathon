"""Suggest bakery thresholds from photos of items you consider GOOD (the 'golden' standard).

  python tools/suggest_thresholds.py samples/golden/
Prints mean +/- 3 sigma ranges. Review them with your bakery's QC person before using.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partguard.config import load_config   # noqa: E402
from partguard import bakery               # noqa: E402

cfg = load_config(ROOT / "configs/bakery.yaml")
rows = []
for f in sorted(Path(sys.argv[1]).glob("*")):
    img = cv2.imread(str(f))
    if img is not None:
        rows += [(r.mean_L, r.std_L, r.roundness, r.solidity, r.diameter_px)
                 for r in bakery.inspect(img, cfg)]
a = np.array(rows)
if len(a) < 10:
    sys.exit(f"only {len(a)} items found - need many more good samples (aim for 50+)")
mu, sd = a.mean(0), a.std(0)
print(f"# from {len(a)} good items")
print("color:")
print(f"  L_min: {mu[0] - 3 * sd[0]:.1f}\n  L_max: {mu[0] + 3 * sd[0]:.1f}\n  max_std_L: {mu[1] + 3 * sd[1]:.1f}")
print("shape:")
print(f"  min_roundness: {max(0.5, mu[2] - 3 * sd[2]):.2f}\n  min_solidity: {max(0.5, mu[3] - 3 * sd[3]):.2f}")
print(f"# equivalent diameter: {mu[4]:.1f} +/- {sd[4]:.1f} px  (convert with px_per_mm)")
