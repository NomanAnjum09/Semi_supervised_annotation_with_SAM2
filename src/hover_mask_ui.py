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
import time
from typing import Tuple, Optional, List
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import ( QLabel, QMessageBox)

from src.helpers import ObjRecord,Click,load_yolo_seg
from src.sam_helper import Sam2Helper


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



