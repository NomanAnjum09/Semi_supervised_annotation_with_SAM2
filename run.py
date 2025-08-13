
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

    def predict_from_point(self, x: int, y: int, positive: bool = True) -> Tuple[np.ndarray, float]:
        """Return (mask, score). Highest-scoring mask among outputs.
        mask: HxW bool array; score: float in [0,1]
        """
        input_point = np.array([[x, y]])
        input_label = np.array([1 if positive else 0])
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        idx = int(np.argmax(scores))
        return masks[idx].astype(bool), float(scores[idx])


# -------------------- Image Viewer Widget --------------------
class HoverMaskViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self._img_bgr: Optional[np.ndarray] = None          # original image
        self._img_disp: Optional[np.ndarray] = None         # scaled for display (BGR)
        self._disp_qpix: Optional[QPixmap] = None
        self._seg: Optional[Segmenter] = None

        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        # Hover prediction throttle
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(30)  # ms; adjust if needed
        self._hover_timer.timeout.connect(self._do_predict)

        self._last_mouse_pos: Optional[QPoint] = None
        self._last_pred_time = 0.0

        # Last overlay state
        self._last_mask: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

        # Cosmetic
        self._cursor_radius = 3

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
        # Debounce/throttle: restart a short timer; if mouse keeps moving,
        # only predict after the timer fires.
        self._hover_timer.start()

    def leaveEvent(self, e):
        # Clear overlay when cursor leaves
        if self._img_bgr is not None:
            self._last_mask = None
            self._last_score = None
            self._update_pixmap(self._img_disp)
        super().leaveEvent(e)

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

    def _do_predict(self):
        if self._img_bgr is None or self._seg is None or self._last_mouse_pos is None:
            return
        # Small additional throttle (optional)
        now = time.time()
        if now - self._last_pred_time < 0.02:  # 50 FPS max predictions
            return

        pt = self._disp_to_img_coords(self._last_mouse_pos)
        if pt is None:
            # Outside image; clear overlay
            self._last_mask = None
            self._last_score = None
            self._update_pixmap(self._img_disp)
            return

        ix, iy = pt
        try:
            mask, score = self._seg.predict_from_point(ix, iy, positive=True)
            self._last_mask = mask
            self._last_score = score
            self._redraw_overlay(cursor_disp_pt=self._last_mouse_pos)
            self._last_pred_time = now
            # Emit a signal or update parent status bar (via event) if needed
            self.parent().parent().show_score(score)
        except Exception as ex:
            # Show once, then stop predicting to avoid spamming
            self._hover_timer.stop()
            QMessageBox.critical(self, "Prediction Error", str(ex))

    def _redraw_overlay(self, cursor_disp_pt: Optional[QPoint] = None):
        """Blend latest mask onto the display image and draw cursor dot."""
        if self._img_disp is None:
            return

        overlay = self._img_disp.copy()

        if self._last_mask is not None:
            # Resize mask to display size for blending
            mask_disp = cv2.resize(
                self._last_mask.astype(np.uint8),
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # Colorize mask (teal-ish) and alpha-blend
            color = np.array([0, 200, 200], dtype=np.uint8)  # BGR
            alpha = 0.35
            colored = np.zeros_like(overlay)
            colored[mask_disp] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0.0)

            # Outline mask (optional)
            contours, _ = cv2.findContours(mask_disp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)  # thin outline

        # Draw a small cursor dot at the current mouse position for feedback
        if cursor_disp_pt is not None:
            cx, cy = cursor_disp_pt.x(), cursor_disp_pt.y()
            # Ensure it's within the widget bounds
            if 0 <= cx < self.width() and 0 <= cy < self.height():
                cv2.circle(overlay, (cx, cy), self._cursor_radius, (255, 255, 255), -1)

        self._update_pixmap(overlay)

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
