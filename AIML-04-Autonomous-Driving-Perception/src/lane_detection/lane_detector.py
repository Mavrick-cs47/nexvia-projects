"""Classical lane detection: ROI mask, Canny edges, probabilistic Hough, left/right line fit."""
from typing import Optional, Tuple

import cv2
import numpy as np

Line = Tuple[int, int, int, int]


def region_of_interest(edges: np.ndarray) -> np.ndarray:
    h, w = edges.shape[:2]
    polygon = np.array(
        [[(int(0.05 * w), h), (int(0.45 * w), int(0.6 * h)),
          (int(0.55 * w), int(0.6 * h)), (int(0.95 * w), h)]],
        dtype=np.int32,
    )
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def _average_line(segments, h: int) -> Optional[Line]:
    if not segments:
        return None
    slopes, intercepts = zip(*segments)
    slope, intercept = float(np.mean(slopes)), float(np.mean(intercepts))
    if abs(slope) < 1e-6:
        return None
    y1, y2 = h, int(0.6 * h)
    x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
    return (x1, y1, x2, y2)


def detect_lanes(frame: np.ndarray) -> Tuple[Optional[Line], Optional[Line]]:
    """Return (left_line, right_line) as (x1, y1, x2, y2), or None if not found."""
    h = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=100)

    left, right = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 0.4:  # drop near-horizontal segments
                continue
            (left if slope < 0 else right).append((slope, intercept))
    return _average_line(left, h), _average_line(right, h)


def draw_lanes(frame: np.ndarray, left: Optional[Line], right: Optional[Line]) -> np.ndarray:
    out = frame.copy()
    for line in (left, right):
        if line is not None:
            cv2.line(out, line[:2], line[2:], (0, 255, 0), 6)
    return out
