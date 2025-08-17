from typing import Tuple, List, Optional  # add this
import numpy as np
import cv2

from sam2.build_sam import build_sam2, build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Sam2Helper:
    def __init__(self, ckpt_path: str, cfg_path: str, device: Optional[str] = None):
        self.ann_frame_idx = 0
        self.device = device or ("cuda" if self._torch_cuda_available() else "cpu")
        self.kind = None
        self.predictor: Optional[SAM2ImagePredictor] = None
        self.propagator = None
        self._init_model(ckpt_path, cfg_path)

    def _torch_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _init_model(self, ckpt_path: str, config: str):
        try:
            # Propagator (video/camera tracker)
            self.propagator = build_sam2_camera_predictor(config, ckpt_path, device=self.device)
            # Image predictor (hover/click refine)
            sam2 = build_sam2(config, ckpt_path,device=self.device)
            sam2.to(self.device)
            self.predictor = SAM2ImagePredictor(sam2)
            self.kind = "sam2"
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SAM2. Check packages/checkpoint/config. Error: {e}"
            )

    def set_image(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

    def predict_from_points(self, points_xy, labels01) -> Tuple[np.ndarray, float]:
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

    def predict_from_point(self, x: int, y: int, positive: bool = True) -> Tuple[np.ndarray, float]:
        return self.predict_from_points([(x, y)], [1 if positive else 0])

    # --- propagation ---
    def load_first(self, image_bgr: np.ndarray):
        self.propagator.load_first_frame(image_bgr)

    def add_prompts(self, prompts: List[tuple[int, np.ndarray, np.ndarray]]):
        for obj_id, points, labels in prompts:
            self.propagator.add_new_prompt(
                frame_idx=self.ann_frame_idx,
                obj_id=int(obj_id),
                points=points.astype(np.float32),
                labels=labels.astype(np.int32),
            )

    def track(self, image_bgr: np.ndarray):
        return self.propagator.track(image_bgr)
