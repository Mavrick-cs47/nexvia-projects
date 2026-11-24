"""Monocular relative depth with MiDaS (small model by default).

Output is relative inverse depth: larger values are closer. It is not metric distance.
torch is imported lazily so `box_depth` can be used without it.
"""
import cv2
import numpy as np


class MiDaSDepth:
    def __init__(self, model_type: str = "MiDaS_small", device: str = None):
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform if "small" in model_type else transforms.dpt_transform

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a depth map the same size as the frame, normalized to [0, 1]."""
        torch = self._torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            batch = self.transform(rgb).to(self.device)
            pred = self.model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        depth = pred.cpu().numpy()
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)


def box_depth(depth_map: np.ndarray, xyxy) -> float:
    """Median relative depth inside a bounding box."""
    h, w = depth_map.shape
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    return float(np.median(depth_map[y1:y2, x1:x2]))
