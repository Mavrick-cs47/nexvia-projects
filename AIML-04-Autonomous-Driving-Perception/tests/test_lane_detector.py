import cv2
import numpy as np

from src.lane_detection.lane_detector import detect_lanes, draw_lanes


def _synthetic_road(w=1280, h=720):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # left lane marking (slopes up-right) and right lane marking (slopes up-left)
    cv2.line(img, (int(0.25 * w), h), (int(0.46 * w), int(0.62 * h)), (255, 255, 255), 8)
    cv2.line(img, (int(0.75 * w), h), (int(0.54 * w), int(0.62 * h)), (255, 255, 255), 8)
    return img


def test_detects_both_lanes():
    left, right = detect_lanes(_synthetic_road())
    assert left is not None and right is not None
    assert left[0] < right[0]  # left lane starts left of the right lane


def test_blank_frame_returns_none():
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detect_lanes(blank) == (None, None)


def test_draw_lanes_keeps_shape():
    frame = _synthetic_road()
    left, right = detect_lanes(frame)
    out = draw_lanes(frame, left, right)
    assert out.shape == frame.shape
