#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt5 SAM2 UI — Hover to preview masks; right-click add / left-click subtract.
Right panel shows a single-select class list from --classes.

Run:
  python sam2_hover_preview.py \
      --ckpt /path/to/sam2_checkpoint.pt \
      --config configs/sam2.1/sam2.1_hiera_l.yaml \
      --image /path/to/image.jpg \
      --classes A1 A2 A3 A4 A5 A6 B1 B2 B3 B4 B5 B6 C1 C2 C3 C4 C5 C6

Shortcuts:
  Z = undo last click
  C = clear all clicks (return to hover mode)
"""

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
from typing import Tuple, Optional, List

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QFileDialog, QWidget,
    QVBoxLayout, QHBoxLayout, QMessageBox, QAction,
    QListWidget, QAbstractItemView, QFrame
)
from src.helpers import ObjRecord,Click
# -------------------- SAM / SAM2 loader --------------------
class Segmenter:
    def __init__(self, ckpt_path: str, config: str, device: str = None):
        self.device = device or ("cuda" if self._torch_cuda_available() else "cpu")
        self.kind = None  # "sam2"
        self.predictor = None
        self._init_model(ckpt_path, config)

    def _torch_cuda_available(self) -> bool:
        try:
            import torch  # noqa
            return torch.cuda.is_available()
        except Exception:
            return False

    def _init_model(self, ckpt_path: str, config: str):
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
                f"Failed to initialize SAM2. Ensure packages/checkpoint/config are correct. Error: {e}"
            )

    def set_image(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

    # Multi-point (add/subtract) refinement
    def predict_from_points(self, points_xy, labels01) -> Tuple[np.ndarray, float]:
        """
        points_xy: List[(x,y)] or Nx2 array, ORIGINAL image coords
        labels01:  List[int] (1=add, 0=subtract)
        """
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points_xy must be Nx2")
        labs = np.asarray(labels01, dtype=np.int32).reshape(-1)
        if labs.shape[0] != pts.shape[0]:
            raise ValueError("labels01 length must match points")
        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=labs,
            multimask_output=True
        )
        i = int(np.argmax(scores))
        return masks[i].astype(bool), float(scores[i])

    # Optional wrapper for single-point use
    def predict_from_point(self, x: int, y: int, positive: bool = True) -> Tuple[np.ndarray, float]:
        return self.predict_from_points([(x, y)], [1 if positive else 0])


# -------------------- Image Viewer --------------------
class HoverMaskViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.setFocusPolicy(Qt.StrongFocus)

        self._img_bgr: Optional[np.ndarray] = None
        self._img_disp: Optional[np.ndarray] = None
        self._seg: Optional[Segmenter] = None

        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(30)  # ms
        self._hover_timer.timeout.connect(self._do_predict_hover)

        self._last_mouse_pos: Optional[QPoint] = None
        self._last_pred_time = 0.0

        self._last_mask: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

        # SAM-style refinement state
        self._click_points: List[Tuple[int, int]] = []
        self._click_labels: List[int] = []         # 1=add (RMB), 0=subtract (LMB)
        self._show_click_markers = True

        # Class selection (set by MainWindow)
        self._selected_class: Optional[str] = None

        # keep reference to bytes for QImage
        self._qimage_bytes = None
        
        #history
        self._objects: list[ObjRecord] = []

    def set_segmenter(self, seg: Segmenter):
        self._seg = seg

    def set_selected_class(self, name: Optional[str]):
        """MainWindow calls this when selection changes."""
        self._selected_class = name

    def set_image(self, img_bgr: np.ndarray):
        assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "Expected BGR image"
        self._img_bgr = img_bgr.copy()
        self._prepare_display_image()
        if self._seg:
            self._seg.set_image(self._img_bgr)
        # clear pending and history
        self._last_mask = None
        self._last_score = None
        self._click_points.clear()
        self._click_labels.clear()
        self._objects.clear()
        self._update_pixmap(self._img_disp)


    def _prepare_display_image(self):
        if self._img_bgr is None:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        H, W = self._img_bgr.shape[:2]
        scale = min(w / W, h / H)
        scale = max(scale, 1e-6)
        newW, newH = int(W * scale), int(H * scale)
        disp = cv2.resize(self._img_bgr, (newW, newH), interpolation=cv2.INTER_AREA)
        self._img_disp = disp
        self._scale = scale
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
        # Always run hover preview (even if we already have clicks for this object)
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

        self._click_points.append(pt_img)
        self._click_labels.append(label)
        try:
            mask, score = self._seg.predict_from_points(self._click_points, self._click_labels)
            self._last_mask = mask
            self._last_score = score
            self._redraw_overlay()
            mw = self.window()
            if hasattr(mw, "show_score"):
                mw.show_score(score)
        except Exception as ex:
            QMessageBox.critical(self, "Prediction Error", str(ex))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Z:
            if self._click_points:
                self._click_points.pop()
                self._click_labels.pop()
                self._recompute_from_clicks_or_clear()
        elif e.key() == Qt.Key_C:
            self._click_points.clear()
            self._click_labels.clear()
            self._last_mask = None
            self._last_score = None
            self._redraw_overlay()
        else:
            super().keyPressEvent(e)

    def _recompute_from_clicks_or_clear(self):
        if self._click_points:
            try:
                mask, score = self._seg.predict_from_points(self._click_points, self._click_labels)
                self._last_mask = mask
                self._last_score = score
                self._redraw_overlay()
                mw = self.window()
                if hasattr(mw, "show_score"):
                    mw.show_score(score)
            except Exception as ex:
                QMessageBox.critical(self, "Prediction Error", str(ex))
        else:
            self._last_mask = None
            self._last_score = None
            self._redraw_overlay()
    
        # ---- history / commit API ----
    def has_pending(self) -> bool:
        """Do we have a mask-in-progress (clicks recorded but not saved)?"""
        return len(self._click_points) > 0

    def commit_current_object(self, class_name: str) -> Optional[ObjRecord]:
        """
        Save the current refined object (clicks + labels + score) under class_name,
        clear pending state, and return the saved record.
        """
        if not self._click_points:
            return None

        # Ensure we have the latest score
        if self._last_score is None:
            try:
                _, score = self._seg.predict_from_points(self._click_points, self._click_labels)
                self._last_score = score
            except Exception:
                self._last_score = 0.0

        clicks = [Click(x=xy[0], y=xy[1], label=lb)
                  for xy, lb in zip(self._click_points, self._click_labels)]
        rec = ObjRecord(class_name=class_name,
                        clicks=clicks,
                        score=float(self._last_score or 0.0))
        self._objects.append(rec)

        # Clear pending for next object
        self._click_points.clear()
        self._click_labels.clear()
        self._last_mask = None
        self._last_score = None
        self._redraw_overlay()
        return rec

    def get_object_count(self) -> int:
        return len(self._objects)

    def get_history(self) -> list[ObjRecord]:
        """Return a copy of the saved objects list."""
        return list(self._objects)


    def leaveEvent(self, e):
        if self._img_bgr is not None and not self._click_points:
            self._last_mask = None
            self._last_score = None
            self._update_pixmap(self._img_disp)
        super().leaveEvent(e)

    def _disp_to_img_coords(self, p: QPoint) -> Optional[Tuple[int, int]]:
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

    def _do_predict_hover(self):
        if self._img_bgr is None or self._seg is None or self._last_mouse_pos is None:
            return

        # Throttle
        now = time.time()
        if now - self._last_pred_time < 0.02:
            return

        # Map cursor to ORIGINAL image coords
        pt = self._disp_to_img_coords(self._last_mouse_pos)
        if pt is None:
            self._last_mask = None
            self._last_score = None
            self._update_pixmap(self._img_disp)
            return

        # Combine existing clicks + hover point (hover = positive probe)
        points_xy = list(self._click_points) + [pt]
        labels01  = list(self._click_labels) + [1]

        try:
            mask, score = self._seg.predict_from_points(points_xy, labels01)
            self._last_mask = mask
            self._last_score = score
            self._redraw_overlay()
            self._last_pred_time = now

            mw = self.window()
            if hasattr(mw, "show_score"):
                mw.show_score(score)
        except Exception as ex:
            self._hover_timer.stop()
            QMessageBox.critical(self, "Prediction Error", str(ex))


    def _redraw_overlay(self):
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

            # Colorize mask and alpha-blend
            color = np.array([0, 200, 200], dtype=np.uint8)  # BGR
            alpha = 0.35
            colored = np.zeros_like(overlay)
            colored[mask_disp] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0.0)

            # Outline
            contours, _ = cv2.findContours(mask_disp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

        # Draw clicked points (add=green, subtract=red), mapped onto the SCALED image (no widget offsets)
        if self._show_click_markers and self._click_points:
            H, W = overlay.shape[:2]
            for (x, y), lab in zip(self._click_points, self._click_labels):
                dx = int(round(x * self._scale))
                dy = int(round(y * self._scale))
                dx = max(0, min(W - 1, dx))
                dy = max(0, min(H - 1, dy))
                color = (0, 255, 0) if lab == 1 else (0, 0, 255)
                cv2.circle(overlay, (dx, dy), 4, color, 2)

        self._update_pixmap(overlay)

    def _update_pixmap(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])
        bytes_per_line = rgb.strides[0]

        # Use tobytes() for broad PyQt compatibility (avoids memoryview TypeError)
        self._qimage_bytes = rgb.tobytes()
        qimg = QImage(self._qimage_bytes, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix)


# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self, seg: Segmenter, img_bgr: np.ndarray, classes: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM2 Hover + Click-Refine — PyQt5")

        self.viewer = HoverMaskViewer(self)
        self.viewer.set_segmenter(seg)
        self.viewer.set_image(img_bgr)

        # ---- right-side class list ----
        self.class_list = QListWidget()
        self.class_list.addItems(classes)
        self.class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.viewer.set_selected_class(self.class_list.currentItem().text())
        self.class_list.currentTextChanged.connect(self.on_class_changed)
        self.class_list.setMinimumHeight(340)
        side = QFrame()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("Classes"))
        side_layout.addWidget(self.class_list)
        side.setFixedWidth(220)


        # Central layout: image viewer (stretch) + side panel
        central = QWidget(self)
        h = QHBoxLayout(central)
        h.addWidget(self.viewer, stretch=1)
        h.addWidget(side)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.status.showMessage("Right-click=Add, Left-click=Subtract, Z=Undo, C=Clear. Pick a class on the right.")

        # Menu: File -> Open Image
        act_open = QAction("Open Image…", self)
        act_open.triggered.connect(self.open_image)
        self.menuBar().addMenu("&File").addAction(act_open)

        self.resize(1280, 800)

    def on_class_changed(self, name: str):
        self.viewer.set_selected_class(name)
        self.show_message(f"Selected class: {name}")

    def show_score(self, s: Optional[float]):
        if s is None:
            self.status.clearMessage()
        else:
            self.status.showMessage(f"Mask score: {s:.3f}")

    def show_message(self, msg: str):
        self.status.showMessage(msg)

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

    w = MainWindow(seg, img_bgr, classes=args.classes)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
