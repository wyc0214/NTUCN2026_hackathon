"""Add reference crops to refs/ from a photo whose contents you KNOW.

  python tools/enroll_bento.py photo.jpg rice=rice main=fried_chicken side1=greens side2=egg
  python tools/enroll_bento.py empty_tray.jpg rice=empty main=empty side1=empty side2=empty

Take 5-10 photos per dish under the SAME lighting/camera as production. More variety = fewer false alarms.
"""
import sys
import time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partguard.config import load_config   # noqa: E402
from partguard.bento import crop_roi       # noqa: E402

img = cv2.imread(sys.argv[1])
cfg = load_config(ROOT / "configs/bento.yaml")["bento"]
for pair in sys.argv[2:]:
    comp, label = pair.split("=")
    crop, _ = crop_roi(img, cfg["tray"]["compartments"][comp], cfg["tray"].get("inset", 0.08))
    d = ROOT / "refs" / label
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{int(time.time() * 1000)}_{comp}.png"), crop)
    print(f"saved {label} <- {comp}")
