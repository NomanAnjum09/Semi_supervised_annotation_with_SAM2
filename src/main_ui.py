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

from typing import Optional

import numpy as np
import cv2

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QFileDialog, QWidget,
    QVBoxLayout, QHBoxLayout, QMessageBox, QAction,
    QListWidget, QAbstractItemView, QFrame, QListWidgetItem, QPushButton
)
from PyQt5.QtWidgets import QProgressDialog
from src.helpers import mask_to_yolo_lines, merge_yolo_seg
from src.sam_helper import Sam2Helper
from src.hover_mask_ui import HoverMaskViewer
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
        wrote = self._propagate_blocking(helper, frames, obj_id_to_class_id)

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


    def _propagate_blocking(self, helper, frames: list[str], obj_id_to_class_id: dict[int, int]) -> int:
        """
        Runs propagation synchronously with a modal progress dialog.
        Blocks the UI until finished. Returns number of frames written.
        """
        total = len(frames)
        dlg = QProgressDialog("Starting…", None, 0, total, self)
        dlg.setWindowTitle("Propagating")
        dlg.setWindowModality(Qt.ApplicationModal)     # block UI
        dlg.setCancelButton(None)                      # no cancel button
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        # Optional: wait cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

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
        try:
            with ctx:
                for idx, path in enumerate(frames, 1):
                    # update progress text
                    dlg.setLabelText(f"Processing {os.path.basename(path)} ({idx}/{total})…")
                    dlg.setValue(idx - 1)             # show progress before work
                    QApplication.processEvents()      # repaint dialog while blocking UI

                    frame = cv2.imread(path, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    try:
                        out_obj_ids, out_mask_logits = helper.track(frame)
                    except Exception as ex:
                        # Non-fatal: skip this frame
                        self.show_message(f"Tracking failed on {os.path.basename(path)}: {ex}")
                        continue

                    # Convert logits -> YOLO lines
                    yolo_lines = []
                    for i, obj_id in enumerate(out_obj_ids):
                        logits = out_mask_logits[i]
                        if hasattr(logits, "shape") and getattr(logits, "ndim", 0) == 3:
                            m = (logits[0] > 0).detach().cpu().numpy().astype(np.uint8)
                        else:
                            m = (logits > 0).detach().cpu().numpy().astype(np.uint8)
                        class_id = obj_id_to_class_id.get(int(obj_id), 0)
                        yolo_lines.extend(mask_to_yolo_lines(m, class_id))

                    # Merge (append without overwriting)
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    merge_yolo_seg(txt_path, yolo_lines, dedup=True)
                    wrote += 1

                    dlg.setValue(idx)                 # advance after work
                    QApplication.processEvents()

            dlg.setValue(total)
        finally:
            dlg.close()
            QApplication.restoreOverrideCursor()

        return wrote
