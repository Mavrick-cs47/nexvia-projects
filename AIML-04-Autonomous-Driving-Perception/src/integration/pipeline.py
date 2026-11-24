"""Per-frame perception pipeline: detection + tracking + lanes (+ optional depth)."""
import argparse

import cv2
import numpy as np

from src.lane_detection.lane_detector import detect_lanes, draw_lanes


class PerceptionPipeline:
    def __init__(self, weights: str, use_depth: bool = False, conf: float = 0.25):
        from src.tracking.bytetrack_tracker import ByteTrackTracker

        self.tracker = ByteTrackTracker(weights, conf=conf)
        self.depth = None
        if use_depth:
            from src.depth_estimation.midas_depth import MiDaSDepth

            self.depth = MiDaSDepth()

    def process_frame(self, frame: np.ndarray):
        """Return (annotated_frame, tracks). Each track may carry a relative depth value."""
        from src.depth_estimation.midas_depth import box_depth

        tracks = self.tracker.update(frame)
        left, right = detect_lanes(frame)
        out = draw_lanes(frame, left, right)

        depth_map = self.depth.predict(frame) if self.depth is not None else None
        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.xyxy)
            label = f"{self.tracker.names[t.cls]} #{t.track_id}"
            if depth_map is not None:
                label += f" d={box_depth(depth_map, t.xyxy):.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out, tracks


def main():
    p = argparse.ArgumentParser(description="Run the perception pipeline on a video")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--out", default="results/output.mp4")
    p.add_argument("--depth", action="store_true", help="enable MiDaS depth (slower)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pipe = PerceptionPipeline(args.weights, use_depth=args.depth)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, _ = pipe.process_frame(frame)
        writer.write(annotated)
    cap.release()
    writer.release()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
