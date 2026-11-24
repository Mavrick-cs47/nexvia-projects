"""ByteTrack multi-object tracking through the Ultralytics tracker interface."""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Track:
    track_id: int
    cls: int
    conf: float
    xyxy: tuple


class ByteTrackTracker:
    def __init__(self, weights: str = "weights/best.pt", conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def update(self, frame: np.ndarray) -> List[Track]:
        """Run detection and tracking on one BGR frame. Tracker state persists across calls."""
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        return [Track(i, c, s, tuple(b)) for i, c, s, b in zip(ids, cls, conf, xyxy)]
