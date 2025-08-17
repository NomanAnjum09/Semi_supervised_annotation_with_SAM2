

import os, os.path as osp
os.environ.pop("QT_PLUGIN_PATH", None)                 # avoid cv2's plugin dir
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")        # prefer X11 on Linux
try:
    import PyQt5  # import first to locate its plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = osp.join(
        osp.dirname(PyQt5.__file__), "Qt", "plugins", "platforms"
    )
except Exception:
    pass

import sys
import argparse
import cv2
from PyQt5.QtWidgets import (QApplication)
from src.helpers import list_image_paths
from src.sam_helper import Sam2Helper
from src.main_ui import MainWindow
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images", help="Folder with images to annotate")
    ap.add_argument("--checkpoint", default="models/sam2.1_hiera_large.pt", help="Path to SAM2/SAM checkpoint .pth")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 (e.g., sam2_hiera_l) or SAM (e.g., vit_h)")
    ap.add_argument("--classes", nargs="*", default=["A1","A2","A3","A4","A5","A6","B1","B2","B3","B4","B5",
                                                     "B6","C1","C2","C3","C4","C5","C6"], help="Initial class names")
    return ap.parse_args()

def main():
    args = parse_args()

    # Resolve image list from folder
    image_paths = list_image_paths(args.images)
    if not image_paths:
        print(f"ERROR: No images found in folder: {args.images}", file=sys.stderr)
        sys.exit(2)

    # Build segmenter
    try:
        seg = Sam2Helper(args.checkpoint, args.config)
    except Exception as e:
        print(f"[Sam2Helper Init Error] {e}", file=sys.stderr)
        sys.exit(3)

    app = QApplication(sys.argv)

    # Load only the FIRST image initially (on demand for others)
    first_path = image_paths[0]
    img_bgr = cv2.imread(first_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"ERROR: Failed to read image: {first_path}", file=sys.stderr)
        sys.exit(2)

    w = MainWindow(seg, img_bgr, classes=args.classes,
                   images_dir=args.images, image_paths=image_paths)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
