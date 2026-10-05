"""Draw the compartments of YOUR tray on a real photo and print the YAML for configs/bento.yaml.

  python tools/select_rois.py photo.jpg rice main side1 side2
For each name, drag a rectangle and press ENTER (or SPACE). Press c to cancel one.
"""
import sys
import cv2

img = cv2.imread(sys.argv[1])
h, w = img.shape[:2]
print("      compartments:")
for name in sys.argv[2:]:
    x, y, rw, rh = cv2.selectROI(f"select '{name}'", img, showCrosshair=False)
    cv2.destroyAllWindows()
    print(f"        {name}: [{x / w:.3f}, {y / h:.3f}, {rw / w:.3f}, {rh / h:.3f}]")
