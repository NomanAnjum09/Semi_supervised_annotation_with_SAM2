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
    QListWidget, QAbstractItemView, QFrame, QListWidgetItem, QPushButton
)

from src.helpers import ObjRecord,Click, list_image_paths, load_yolo_seg, mask_to_yolo_lines, merge_yolo_seg
from src.sam_helper import Sam2Helper
# -------------------- SAM / SAM2 loader --------------------

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
        self._seg: Optional[Sam2Helper] = None

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
        self._editing_original: Optional[ObjRecord] = None

    def set_segmenter(self, seg: Sam2Helper):
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
        self._editing_original = None
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
        ix, iy = pt_img

        # If user clicks on a FROZEN object, pull it back into edit mode.
        obj_idx = self._hit_test_object(ix, iy)
        if obj_idx is not None:
            obj = self._objects.pop(obj_idx)  # remove from frozen so it won't double-draw
            
            # Restore its clicks/labels so user can refine
            self._click_points = [(c.x, c.y) for c in obj.clicks]
            self._click_labels = [c.label for c in obj.clicks]
            self._last_mask = obj.mask.astype(bool)
            self._last_score = obj.score
            self._editing_original = obj
            # Do NOT treat this click as an add/subtract action; just enter edit mode
            self._redraw_overlay()
            mw = self.window()
            if hasattr(mw, "update_saved_count"):
                mw.update_saved_count()
            if hasattr(mw, "show_message"):
                mw.show_message(f"Editing previously saved object (class '{obj.class_name}').")
            if hasattr(mw, "set_selected_class_name"):
                mw.set_selected_class_name(obj.class_name, announce=True)
            return

        # Otherwise, normal add/subtract behavior on current (pending) object
        if e.button() == Qt.LeftButton:
            label = 1  # add
        elif e.button() == Qt.RightButton:
            label = 0  # sub
        else:
            return

        self._click_points.append((ix, iy))
        self._click_labels.append(label)
        try:
            mask, score = self._seg.predict_from_points(self._click_points, self._click_labels)
            self._last_mask = mask
            self._last_score = float(score)
            self._redraw_overlay()
            mw = self.window()
            if hasattr(mw, "show_score"):
                mw.show_score(self._last_score)
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
        elif e.key() == Qt.Key_Escape:
            # Clear everything except the committed (frozen) objects
            self.cancel_edit_or_clear_pending()
            # also clear class selection highlight:
            mw = self.window()
            if hasattr(mw, "class_list"):
                mw.class_list.clearSelection()
        elif e.key() in (Qt.Key_Delete,):
    # If we’re editing a committed object, delete it
            if self._editing_original is not None:
                self.delete_editing_object()
            else:
                if self._click_points:
                    self._click_points.pop()
                    self._click_labels.pop()
                    self._recompute_from_clicks_or_clear()
        else:
            super().keyPressEvent(e)

    def cancel_edit_or_clear_pending(self):
        """
        If editing a previously committed object, restore it unchanged and exit edit mode.
        Otherwise, clear the in-progress (uncommitted) clicks/mask/score.
        """
        if self._editing_original is not None:
            # Put the original back exactly as it was
            self._objects.append(self._editing_original)
            restored_class = self._editing_original.class_name
            self._editing_original = None
            
            # Clear any in-progress edits
            self._click_points.clear()
            self._click_labels.clear()
            self._last_mask = None
            self._last_score = None

            self._redraw_overlay()

            # (Optional) highlight its class again
            mw = self.window()
            if hasattr(mw, "update_saved_count"):
                mw.update_saved_count()
            if hasattr(mw, "set_selected_class_name"):
                mw.set_selected_class_name(restored_class, announce=False)
        else:
            # No edit-in-progress; just clear pending (keeps committed visible)
            self.clear_pending()

    def delete_editing_object(self):
        """
        If we're editing a previously committed object (after clicking it),
        delete it permanently (do NOT restore it), and clear any pending state.
        """
        if self._editing_original is None:
            return  # nothing to delete

        deleted_class = self._editing_original.class_name
        # The object was already popped from _objects when we entered edit mode.
        self._editing_original = None

        # Clear any in-progress points/mask/score
        self._click_points.clear()
        self._click_labels.clear()
        self._last_mask = None
        self._last_score = None

        # Redraw shows only the remaining committed objects
        self._redraw_overlay()

        # Update UI counters/messages if available
        mw = self.window()
        if hasattr(mw, "update_saved_count"):
            mw.update_saved_count()
        if hasattr(mw, "show_message"):
            mw.show_message(f"Deleted committed object ('{deleted_class}').")


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
    def clear_pending(self):
        """
        Clear any in-progress (uncommitted) clicks/mask/score,
        but keep the committed objects intact.
        """
        self._click_points.clear()
        self._click_labels.clear()
        self._last_mask = None
        self._last_score = None
        self._editing_original = None
        self._redraw_overlay()

        # Optionally tell the MainWindow to update UI
        mw = self.window()
        if hasattr(mw, "show_score"):
            mw.show_score(None)

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
        self._editing_original = None

    # Freeze: clear pending so a new object can begin
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
    # Clear ONLY the hover/pending preview; keep frozen objects visible.
        if self._img_bgr is not None:
            self._last_mask = None
            self._last_score = None
            self._redraw_overlay()   # <- draws frozen objects
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
    
    def _hit_test_object(self, ix: int, iy: int) -> Optional[int]:
        """Return index of the topmost frozen object whose mask contains (ix, iy); else None."""
        if not self._objects:
            return None
        # Iterate from last to first so the most-recently saved object is "on top"
        for idx in range(len(self._objects) - 1, -1, -1):
            obj = self._objects[idx]
            m = obj.mask
            # Bounds check (should match original image size)
            if 0 <= iy < m.shape[0] and 0 <= ix < m.shape[1]:
                if bool(m[iy, ix]):
                    return idx
        return None

    def commit_current_object(self, class_name: str) -> Optional[ObjRecord]:
        """
        Save the current refined object (clicks + labels + score + mask) under class_name,
        clear pending state, and return the saved record. The saved object remains visible (frozen).
        """
        if not self._click_points:
            return None

        # Compute final mask + score from current clicks
        try:
            mask, score = self._seg.predict_from_points(self._click_points, self._click_labels)
            self._last_mask = mask
            self._last_score = float(score)
        except Exception:
            if self._last_mask is None:
                return None
            # Score may be None if something failed; default to 0.0
            self._last_score = float(self._last_score or 0.0)

        clicks = [Click(x=xy[0], y=xy[1], label=lb)
                for xy, lb in zip(self._click_points, self._click_labels)]

        rec = ObjRecord(
            class_name=class_name,
            clicks=clicks,
            score=float(self._last_score or 0.0),
            mask=self._last_mask.astype(bool)
        )
        self._objects.append(rec)

        # Freeze: clear pending so a new object can begin
        self._click_points.clear()
        self._click_labels.clear()
        self._last_mask = None
        self._last_score = None
        self._redraw_overlay()
        return rec

    def _do_predict_hover(self):
        if self._img_bgr is None or self._seg is None or self._last_mouse_pos is None:
            return

        now = time.time()
        if now - self._last_pred_time < 0.02:
            return

        pt = self._disp_to_img_coords(self._last_mouse_pos)
        if pt is None:
            self._last_mask = None
            self._last_score = None
            self._update_pixmap(self._img_disp)
            return

        # Combine existing clicks + hover probe (1 = positive)
        points_xy = list(self._click_points) + [pt]
        labels01  = list(self._click_labels) + [1]

        try:
            mask, score = self._seg.predict_from_points(points_xy, labels01)
            self._last_mask = mask
            self._last_score = float(score)
            self._redraw_overlay()
            self._last_pred_time = now
            mw = self.window()
            if hasattr(mw, "show_score"):
                mw.show_score(self._last_score)
        except Exception as ex:
            self._hover_timer.stop()
            QMessageBox.critical(self, "Prediction Error", str(ex))



    def _redraw_overlay(self):
        if self._img_disp is None:
            return

        overlay = self._img_disp.copy()

        # --- Draw FROZEN objects (from history) ---
        if self._objects:
            for obj in self._objects:
                mask_disp = cv2.resize(
                    obj.mask.astype(np.uint8),
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                # color overlay for frozen objects (cyan-ish)
                frozen_color = np.array([0, 200, 200], dtype=np.uint8)  # BGR
                frozen_alpha = 0.25
                colored = np.zeros_like(overlay)
                colored[mask_disp] = frozen_color
                overlay = cv2.addWeighted(overlay, 1.0, colored, frozen_alpha, 0.0)
                # outline
                contours, _ = cv2.findContours(mask_disp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

        # --- Draw current (pending) object mask, if any ---
        if self._last_mask is not None:
            mask_disp = cv2.resize(
                self._last_mask.astype(np.uint8),
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            pending_color = np.array([0, 180, 0], dtype=np.uint8)  # greenish for current
            pending_alpha = 0.35
            colored = np.zeros_like(overlay)
            colored[mask_disp] = pending_color
            overlay = cv2.addWeighted(overlay, 1.0, colored, pending_alpha, 0.0)
            # outline
            contours, _ = cv2.findContours(mask_disp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Click markers for the pending object (scaled; no widget offsets)
        if self._show_click_markers and self._click_points:
            H, W = overlay.shape[:2]
            for (x, y), lab in zip(self._click_points, self._click_labels):
                dx = int(round(x * self._scale)); dy = int(round(y * self._scale))
                dx = max(0, min(W - 1, dx)); dy = max(0, min(H - 1, dy))
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

    
    def load_yolo_annotations(self, txt_path: str, class_names: list[str]):
        """
        Load YOLO segmentation (if present) and render as frozen objects.
        Clears any pending/editing state.
        """
        if self._img_bgr is None:
            return
        H, W = self._img_bgr.shape[:2]
        recs = []
        pairs = load_yolo_seg(txt_path)  # list[(class_id, norm_coords)]
        for cls_id, norm in pairs:
            if cls_id < 0 or cls_id >= len(class_names):
                continue
            # to pixel polygon
            pix = np.zeros((norm.shape[0], 2), dtype=np.int32)
            pix[:, 0] = np.clip(np.round(norm[:, 0] * W), 0, W - 1).astype(np.int32)
            pix[:, 1] = np.clip(np.round(norm[:, 1] * H), 0, H - 1).astype(np.int32)

            # rasterize polygon to mask
            m = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(m, [pix.reshape(-1, 1, 2)], 1)
            recs.append(ObjRecord(
                class_name=class_names[cls_id],
                clicks=[],  # unknown from disk
                score=0.0,
                mask=m.astype(bool)
            ))

        # replace history with loaded objects
        self._objects = recs
        self._click_points.clear()
        self._click_labels.clear()
        self._last_mask = None
        self._last_score = None
        self._editing_original = None
        self._redraw_overlay()



# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self, seg: Sam2Helper, img_bgr: np.ndarray, classes: list, parent=None,
                 images_dir: Optional[str] = None, image_paths: Optional[list] = None):
        super().__init__(parent)
        self.setWindowTitle("SAM2 Hover + Click-Refine — PyQt5")

        self._images_dir = images_dir
        self._image_paths = image_paths or []
        self._current_image_path: Optional[str] = None

        self.viewer = HoverMaskViewer(self)
        self.viewer.set_segmenter(seg)
        self.viewer.set_image(img_bgr)

        # ---- right-side class list ----
        self.class_list = QListWidget()
        self.class_list.addItems(classes)
        self.class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.class_list.clearSelection()
        self.class_list.itemClicked.connect(self.on_class_clicked)
        self.class_list.setMinimumHeight(220)

        # ---- images list (loaded on demand) ----
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.itemClicked.connect(self.on_image_clicked)
        # Populate only filenames, store full path in item data
        for p in (self._image_paths or []):
            item_text = osp.basename(p)
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, p)
            self.image_list.addItem(item)
        self.propagate_btn = QPushButton("Propagate")
        self.propagate_btn.clicked.connect(self.on_propagate_clicked)
        side = QFrame()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("Classes"))
        side_layout.addWidget(self.class_list)
        self.saved_label = QLabel("Saved: 0")
        side_layout.addWidget(self.saved_label)
        side_layout.addWidget(QLabel("Images"))
        side_layout.addWidget(self.image_list)
        side_layout.addWidget(self.propagate_btn)

        side.setFixedWidth(260)

        # Central layout: image viewer (stretch) + side panel
        central = QWidget(self)
        h = QHBoxLayout(central)
        h.addWidget(self.viewer, stretch=1)
        h.addWidget(side)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.status.showMessage(
            "Hover previews (+pending). Right-click=Add, Left-click=Subtract, Z=Undo, C=Clear. "
            "Click a class to SAVE. Click a frozen object to EDIT."
        )

        # Menu: File -> Open Image
        act_open = QAction("Open Image…", self)
        act_open.triggered.connect(self.open_image)
        self.menuBar().addMenu("&File").addAction(act_open)

        self.resize(1280, 800)

        # If we have a list of images, select the first (loads on demand)
        if self._image_paths:
            self.select_image_index(0)

    # -------- image loading on demand --------
    def select_image_index(self, idx: int):
        if idx < 0 or idx >= self.image_list.count():
            return
        item = self.image_list.item(idx)
        if item is None:
            return
        self.image_list.setCurrentRow(idx)
        path = item.data(Qt.UserRole)
        self._load_image_path(path)

    def on_image_clicked(self, item):
        path = item.data(Qt.UserRole)
        self._load_image_path(path)

    def _load_image_path(self, path: str):
        if not path or not osp.isfile(path):
            QMessageBox.critical(self, "Error", f"Image not found:\n{path}")
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to read image:\n{path}")
            return

        self.viewer.set_image(img)  # resets pending + history
        self._current_image_path = path

        # Load YOLO labels if present
        txt_path = osp.splitext(path)[0] + ".txt"
        try:
            self.viewer.load_yolo_annotations(txt_path, [self.class_list.item(i).text() for i in range(self.class_list.count())])
        except Exception as ex:
            # Non-fatal: just show a message
            self.show_message(f"Note: failed to load labels for {path}: {ex}")

        self.update_saved_count()
        self.show_message(f"Loaded: {path}")


    def on_propagate_clicked(self):
        # Basic validation
        if not self._image_paths:
            QMessageBox.warning(self, "Propagate", "No images in the folder.")
            return
        if self._current_image_path is None:
            QMessageBox.warning(self, "Propagate", "Please select an image first.")
            return
        if self.viewer.get_object_count() == 0:
            QMessageBox.information(self, "Propagate", "No saved objects on the current image.")
            return

        helper = self.viewer._seg  # Sam2Helper
        if helper is None or helper.propagator is None:
            QMessageBox.critical(self, "Propagate", "SAM2 propagator is not initialized.")
            return

        # Seed prompts
        prompts, obj_id_to_class_id = self._collect_prompts_from_saved_objects()
        if not prompts:
            QMessageBox.information(self, "Propagate", "No prompts with clicks to seed propagation.")
            return

        # Init first frame
        if not self._init_propagation(helper, self._current_image_path):
            return

        # Add prompts
        try:
            helper.add_prompts(prompts)
        except Exception as ex:
            QMessageBox.critical(self, "Propagate", f"Failed to add prompts:\n{ex}")
            return

        # Track across folder and write YOLO
        frames = self._all_frame_paths()
        wrote = self._iterate_and_write_yolo(helper, frames, obj_id_to_class_id)

        self.show_message(f"Propagation complete. Wrote labels for {wrote} images.")
        QMessageBox.information(self, "Propagate", f"Propagation complete.\nWrote labels for {wrote} images.")

    def show_score(self, s: Optional[float]):
        if s is None:
            self.status.clearMessage()
        else:
            self.status.showMessage(f"Mask score: {s:.3f}")

    def on_class_clicked(self, item):
        name = item.text()
        if self.viewer.has_pending():
            rec = self.viewer.commit_current_object(name)
            if rec is not None:
                self.update_saved_count()
                self.class_list.clearSelection()
                self.show_message(f"Saved object #{self.viewer.get_object_count()} as '{name}'.")
        else:
            self.viewer.set_selected_class(name)
            self.show_message(f"Selected class: {name}")

    def update_saved_count(self):
        self.saved_label.setText(f"Saved: {self.viewer.get_object_count()}")

    def set_selected_class_name(self, name: str, announce: bool = True):
        for row in range(self.class_list.count()):
            if self.class_list.item(row).text() == name:
                self.class_list.setCurrentRow(row)
                break
        self.viewer.set_selected_class(name)
        if announce:
            self.show_message(f"Editing object (class '{name}')")

    def show_message(self, msg: str):
        self.status.showMessage(msg)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        self._load_image_path(path)
    
    # --- helpers: UI/data access ---

    def _get_class_names(self) -> list[str]:
        return [self.class_list.item(i).text() for i in range(self.class_list.count())]

    def _class_id_of(self, class_names: list[str], name: str) -> int:
        try:
            return class_names.index(name)
        except ValueError:
            return 0

    def _all_frame_paths(self) -> list[str]:
        # already sorted when created; keep a defensive sort here
        return sorted([p for p in (self._image_paths or []) if os.path.isfile(p)])
    
    # --- helpers: propagation pipeline ---

    def _collect_prompts_from_saved_objects(self) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], dict[int, int]]:
        """
        Returns:
        prompts: list of (obj_id, points Nx2 float32, labels N int32)
        obj_id_to_class_id: {obj_id -> class_id}
        Uses only objects that have 'clicks' (user prompts).
        """
        saved_objs = self.viewer.get_history()
        class_names = self._get_class_names()

        prompts = []
        obj_id_to_class_id = {}
        next_id = 1

        for rec in saved_objs:
            if not rec.clicks:
                continue
            pts = np.array([(c.x, c.y) for c in rec.clicks], dtype=np.float32)
            lbs = np.array([c.label for c in rec.clicks], dtype=np.int32)
            prompts.append((next_id, pts, lbs))
            obj_id_to_class_id[next_id] = self._class_id_of(class_names, rec.class_name)
            next_id += 1

        return prompts, obj_id_to_class_id


    def _init_propagation(self, helper, first_image_path: str) -> bool:
        """
        Loads first frame into propagator. Returns True on success, False on failure (and shows UI message).
        """
        img = cv2.imread(first_image_path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Propagate", f"Failed to read: {first_image_path}")
            return False
        try:
            helper.load_first(img)
            return True
        except Exception as ex:
            QMessageBox.critical(self, "Propagate", f"Failed to load first frame:\n{ex}")
            return False


    def _track_one_frame(self, helper, frame_bgr: np.ndarray):
        """
        Wraps helper.track(frame_bgr) with a small try/except. Returns (out_obj_ids, out_mask_logits) or (None, None).
        """
        try:
            return helper.track(frame_bgr)
        except Exception as ex:
            return None, ex


    def _yolo_lines_from_logits_list(self, out_obj_ids, out_mask_logits, obj_id_to_class_id: dict[int, int]) -> list[str]:
        """
        Converts SAM2 logits to YOLO-seg lines for a single frame.
        """
        yolo_lines = []
        for i, obj_id in enumerate(out_obj_ids):
            logits = out_mask_logits[i]
            # Expect CxHxW; take first channel
            if hasattr(logits, "shape") and getattr(logits, "ndim", 0) == 3:
                m = (logits[0] > 0).detach().cpu().numpy().astype(np.uint8)
            else:
                m = (logits > 0).detach().cpu().numpy().astype(np.uint8)
            class_id = obj_id_to_class_id.get(int(obj_id), 0)
            yolo_lines.extend(mask_to_yolo_lines(m, class_id))
        return yolo_lines


    def _iterate_and_write_yolo(self, helper, frames: list[str], obj_id_to_class_id: dict[int, int]) -> int:
        """
        Iterates over frame paths, tracks, and writes YOLO labels. Returns count of frames written.
        """
        # Optional autocast for CUDA
        use_cuda = False
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            pass

        if use_cuda:
            import torch
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        wrote = 0
        with ctx:
            for path in frames:
                frame = cv2.imread(path, cv2.IMREAD_COLOR)
                if frame is None:
                    self.show_message(f"Skip unreadable: {os.path.basename(path)}")
                    continue

                out_obj_ids, out_mask_logits_or_err = self._track_one_frame(helper, frame)
                if out_obj_ids is None:
                    self.show_message(f"Tracking failed on {os.path.basename(path)}: {out_mask_logits_or_err}")
                    continue

                yolo_lines = self._yolo_lines_from_logits_list(out_obj_ids, out_mask_logits_or_err, obj_id_to_class_id)
                txt_path = os.path.splitext(path)[0] + ".txt"
                merge_yolo_seg(txt_path, yolo_lines, dedup=True)
                wrote += 1

        return wrote




# -------------------- CLI / Entrypoint --------------------
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
