
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
import time
from typing import Tuple, Optional

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QMessageBox, QAction
)

# -------------------- SAM / SAM2 loader (as provided) --------------------
class Segmenter:
    def __init__(self, ckpt_path: str, config: str, device: str = None):
        self.device = device or ("cuda" if self._torch_cuda_available() else "cpu")
        self.kind = None  # "sam2" or "sam"
        self.predictor = None
        self._init_model(ckpt_path, config)

    def _torch_cuda_available(self) -> bool:
        try:
            import torch  # noqa
            return torch.cuda.is_available()
        except Exception:
            return False

    def _init_model(self, ckpt_path: str, config: str):
        # Try SAM2 first
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2 = build_sam2(config, ckpt_path)
            sam2.to(self.device)
            self.predictor = SAM2ImagePredictor(sam2)
            self.kind = "sam2"
            return
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SAM2 or SAM. Ensure packages/checkpoint/model_type are correct. Error: {e}"
            )

    def set_image(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

    def predict_from_points(self, points_xy, labels01) -> Tuple[np.ndarray, float]:
        """
        points_xy: List[Tuple[int,int]] or Nx2 np.array in image coords (original size)
        labels01:  List[int] or (N,) np.array with 1 (add) or 0 (subtract)
        Returns: (best_mask_bool, best_score_float)
        """
        pts = np.asarray(points_xy, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points_xy must be Nx2")
        labs = np.asarray(labels01, dtype=np.int32).reshape(-1)
        if labs.shape[0] != pts.shape[0]:
            raise ValueError("labels01 length must match points")
        masks, scores, _ = self.predictor.predict(
            point_coords=pts.astype(np.float32),
            point_labels=labs.astype(np.int32),
            multimask_output=True
        )
        i = int(np.argmax(scores))
        return masks[i].astype(bool), float(scores[i])
    
    




# -------------------- Image Viewer Widget --------------------
class HoverMaskViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setContextMenuPolicy(Qt.NoContextMenu)  # let right-click be our add-click

        self._img_bgr = None
        self._img_disp = None
        self._seg = None

        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(30)
        self._hover_timer.timeout.connect(self._do_predict_hover)

        self._last_mouse_pos = None
        self._last_pred_time = 0.0

        self._last_mask = None
        self._last_score = None

        # NEW: click history for SAM-style refinement
        self._click_points = []   # List[(x,y)]
        self._click_labels = []   # List[int] 1=add, 0=subtract

        # Optional: draw small markers at clicked points
        self._show_click_markers = True
    
    def _disp_to_img_coords(self, p: QPoint) -> Optional[Tuple[int, int]]:
        """Map widget/display coords back to original image coords."""
        if self._img_bgr is None:
            return None
        x = p.x() - self._offset_x
        y = p.y() - self._offset_y
        if x < 0 or y < 0:
            return None
        H, W = self._img_bgr.shape[:2]
        ix = int(x / self._scale)
        iy = int(y / self._scale)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return None
        return ix, iy

    def _update_pixmap(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Make sure memory is contiguous and uint8
        rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

        # bytesPerLine = stride for the first dimension
        bytes_per_line = rgb.strides[0]

        # Build QImage from the NumPy buffer
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Keep a reference so the data pointer stays valid
        self._qimage_buf = rgb

        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix)


    def set_segmenter(self, seg: Segmenter):
        self._seg = seg

    def set_image(self, img_bgr: np.ndarray):
        assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "Expected BGR image"
        self._img_bgr = img_bgr.copy()
        self._prepare_display_image()
        if self._seg:
            self._seg.set_image(self._img_bgr)
        self._last_mask = None
        self._last_score = None
        self._update_pixmap(self._img_disp)

    def _prepare_display_image(self):
        """Compute scaled display image to fit the widget while preserving aspect ratio."""
        if self._img_bgr is None:
            return
        # Size to fit current widget
        w = max(1, self.width())
        h = max(1, self.height())
        H, W = self._img_bgr.shape[:2]
        scale = min(w / W, h / H)
        scale = max(scale, 1e-6)
        newW, newH = int(W * scale), int(H * scale)
        disp = cv2.resize(self._img_bgr, (newW, newH), interpolation=cv2.INTER_AREA)
        self._img_disp = disp
        self._scale = scale
        # Compute centering offsets
        self._offset_x = (w - newW) // 2
        self._offset_y = (h - newH) // 2

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._img_bgr is not None:
            self._prepare_display_image()
            self._redraw_overlay()

    def mouseMoveEvent(self, e):
        if self._img_bgr is None or self._seg is None:
            return
        self._last_mouse_pos = e.pos()
        # Only do hover prediction when no refinement clicks yet
        if len(self._click_points) == 0:
            self._hover_timer.start()

    def mousePressEvent(self, e):
        if self._img_bgr is None or self._seg is None:
            return
        pt_img = self._disp_to_img_coords(e.pos())
        if pt_img is None:
            return

        # Left = subtract (0), Right = add (1)
        if e.button() == Qt.LeftButton:
            label = 1
        elif e.button() == Qt.RightButton:
            label = 0
        else:
            return

        # Record click
        self._click_points.append(pt_img)
        self._click_labels.append(label)

        # Predict with all points so far
        try:
            mask, score = self._seg.predict_from_points(self._click_points, self._click_labels)
            self._last_mask = mask
            self._last_score = score
            self._redraw_overlay()
            self.parent().parent().show_score(score)
        except Exception as ex:
            QMessageBox.critical(self, "Prediction Error", str(ex))

    def keyPressEvent(self, e):
        # Z: undo last point, C: clear all points
        if e.key() in (Qt.Key_Z,):
            if self._click_points:
                self._click_points.pop()
                self._click_labels.pop()
                self._recompute_from_clicks_or_clear()
        elif e.key() in (Qt.Key_C,):
            self._click_points.clear()
            self._click_labels.clear()
            self._last_mask = None
            self._last_score = None
            # back to hover mode
            self._redraw_overlay()
        else:
            super().keyPressEvent(e)

    def _recompute_from_clicks_or_clear(self):
        # Recompute current refined mask if any clicks remain; else clear and show base image
        if self._click_points:
            try:
                mask, score = self._seg.predict_from_points(self._click_points, self._click_labels)
                self._last_mask = mask
                self._last_score = score
                self._redraw_overlay()
                self.parent().parent().show_score(score)
            except Exception as ex:
                QMessageBox.critical(self, "Prediction Error", str(ex))
        else:
            self._last_mask = None
            self._last_score = None
            self._redraw_overlay()

    def _do_predict_hover(self):
        # unchanged except: only run when no clicks yet
        if self._img_bgr is None or self._seg is None or self._last_mouse_pos is None:
            return
        if self._click_points:
            return  # refinement in progress; skip hover

        now = time.time()
        if now - self._last_pred_time < 0.02:
            return

        pt = self._disp_to_img_coords(self._last_mouse_pos)
        if pt is None:
            self._last_mask = None
            self._last_score = None
            self._update_pixmap(self._img_disp)
            return

        ix, iy = pt
        try:
            mask, score = self._seg.predict_from_points([(ix, iy)], [1])
            self._last_mask = mask
            self._last_score = score
            self._redraw_overlay()
            self._last_pred_time = now
            self.parent().parent().show_score(score)
        except Exception as ex:
            self._hover_timer.stop()
            QMessageBox.critical(self, "Prediction Error", str(ex))

    def _redraw_overlay(self):
        if self._img_disp is None:
            return

        overlay = self._img_disp.copy()

        if self._last_mask is not None:
            mask_disp = cv2.resize(
                self._last_mask.astype(np.uint8),
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            color = np.array([0, 200, 200], dtype=np.uint8)  # BGR
            alpha = 0.35
            colored = np.zeros_like(overlay)
            colored[mask_disp] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0.0)

            contours, _ = cv2.findContours(mask_disp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

        # Draw clicked points (optional; not the moving cursor)
        if self._show_click_markers and self._click_points:
            H, W = overlay.shape[:2]  # overlay is the scaled image (no margins)
            for (x, y), lab in zip(self._click_points, self._click_labels):
                # Map ORIGINAL image coords -> SCALED image coords (no offsets here)
                dx = int(round(x * self._scale))
                dy = int(round(y * self._scale))
                # clamp to image bounds
                dx = max(0, min(W - 1, dx))
                dy = max(0, min(H - 1, dy))
                color = (0, 255, 0) if lab == 1 else (0, 0, 255)  # add=green, subtract=red
                cv2.circle(overlay, (dx, dy), 4, color, 2)

        self._update_pixmap(overlay)

# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self, seg: Segmenter, img_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM2 Hover Mask Preview — PyQt5")
        self.viewer = HoverMaskViewer(self)
        self.viewer.set_segmenter(seg)
        self.viewer.set_image(img_bgr)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self.viewer)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.status.showMessage("Move the mouse over the image to preview masks.")

        # Menu: File -> Open Image
        act_open = QAction("Open Image…", self)
        act_open.triggered.connect(self.open_image)
        self.menuBar().addMenu("&File").addAction(act_open)

        self.resize(1280, 800)

    def show_score(self, s: Optional[float]):
        if s is None:
            self.status.clearMessage()
        else:
            self.status.showMessage(f"Mask score: {s:.3f}")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to read image:\n{path}")
            return
        try:
            self.viewer.set_image(img)
            self.status.showMessage(f"Loaded: {path}")
        except Exception as ex:
            QMessageBox.critical(self, "Error", str(ex))


# -------------------- CLI / Entrypoint --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images/0.jpg", help="Folder with images to annotate")
    ap.add_argument("--checkpoint", default="models/sam2.1_hiera_large.pt", help="Path to SAM2/SAM checkpoint .pth")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 (e.g., sam2_hiera_l) or SAM (e.g., vit_h)")
    ap.add_argument("--classes", nargs="*", default=["A1","A2","A3","A4","A5","A6","B1","B2","B3","B4","B5",
                                                     "B6","C1","C2","C3","C4","C5","C6"], help="Initial class names")
    return ap.parse_args()

def main():
    args = parse_args()
    # Load image (or ask)
    img_bgr = None
    if args.images:
        img_bgr = cv2.imread(args.images, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"ERROR: Failed to read image: {args.images}", file=sys.stderr)
            sys.exit(2)

    # Build segmenter
    try:
        seg = Segmenter(args.checkpoint, args.config)
    except Exception as e:
        print(f"[Segmenter Init Error] {e}", file=sys.stderr)
        sys.exit(3)

    app = QApplication(sys.argv)

    if img_bgr is None:
        # Show file dialog at startup
        path, _ = QFileDialog.getOpenFileName(None, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            print("No image selected. Exiting.")
            sys.exit(0)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"ERROR: Failed to read image: {path}", file=sys.stderr)
            sys.exit(2)

    w = MainWindow(seg, img_bgr)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
