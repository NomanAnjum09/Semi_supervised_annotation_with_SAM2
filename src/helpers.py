# ---------- Saved-object schema ----------
from dataclasses import dataclass
import numpy as np
import cv2
import os, os.path as osp
from typing import Tuple,Optional
@dataclass
class Click:
    x: int
    y: int
    label: int  # 1 = add, 0 = subtract

@dataclass
class ObjRecord:
    class_name: str
    clicks: list[Click]
    score: float
    mask: np.ndarray


def contours_to_yolo_line(class_id: int, cnt: np.ndarray, w: int, h: int) -> Optional[str]:
    # cnt: Nx1x2 int points. Return None if too small.
    if cnt is None or len(cnt) < 3:
        return None
    xs = cnt[:, 0, 0].astype(np.float32) / max(1, w)
    ys = cnt[:, 0, 1].astype(np.float32) / max(1, h)
    pts = []
    for x, y in zip(xs, ys):
        pts.append(f"{x:.6f}")
        pts.append(f"{y:.6f}")
    return f"{class_id} " + " ".join(pts)

def mask_to_yolo_lines(mask: np.ndarray, class_id: int) -> list[str]:
    """
    mask: HxW (bool or uint8)
    Returns YOLO-seg lines for all external contours in this mask.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    H, W = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in cnts:
        if len(cnt) < 3:
            continue
        # optional simplify
        eps = 0.0025 * (cv2.arcLength(cnt, True) + 1e-6)
        cnt_s = cv2.approxPolyDP(cnt, eps, True)
        line = contours_to_yolo_line(class_id, cnt_s, W, H)
        if line:
            lines.append(line)
    return lines

def load_yolo_seg(txt_path: str) -> list[tuple[int, np.ndarray]]:
    """
    Returns list of (class_id, polygon_pixels Nx2 int).
    """
    out = []
    if not os.path.isfile(txt_path):
        return out
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(parts[0])
                nums = list(map(float, parts[1:]))
                if len(nums) % 2 != 0 or len(nums) < 6:
                    continue
                coords = np.array(nums, dtype=np.float32).reshape(-1, 2)  # normalized
                out.append((cls, coords))
            except Exception:
                continue
    return out

def list_image_paths(folder: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not osp.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        _, ext = osp.splitext(name)
        if ext.lower() in exts:
            files.append(osp.join(folder, name))
    return files

def merge_yolo_seg(txt_path: str, new_lines: list[str], dedup: bool = True):
    """
    Read existing YOLO-seg labels (if any), append new_lines, and write back.
    If dedup=True, exact duplicate lines are removed (string match).
    """
    existing: list[str] = []
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            existing = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not new_lines:
        # Nothing to add; keep existing as-is
        if not os.path.isfile(txt_path):
            # no file → nothing to write
            return
        with open(txt_path, "w", encoding="utf-8") as f:
            for ln in existing:
                f.write(ln + "\n")
        return

    merged = existing + [ln.rstrip("\n") for ln in new_lines if ln.strip()]

    if dedup:
        # exact string-level dedup while preserving order
        seen = set()
        uniq = []
        for ln in merged:
            if ln not in seen:
                seen.add(ln)
                uniq.append(ln)
        merged = uniq

    with open(txt_path, "w", encoding="utf-8") as f:
        for ln in merged:
            f.write(ln + "\n")
