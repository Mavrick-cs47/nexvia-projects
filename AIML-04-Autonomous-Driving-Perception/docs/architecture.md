# Architecture

## Pipeline

```
video frame
   |
   +--> YOLOv8 detector --> boxes, classes, scores
   |          |
   |          +--> ByteTrack --> persistent track IDs
   |
   +--> Lane detector (Canny + Hough) --> left/right lane lines
   |
   +--> MiDaS (optional) --> relative inverse-depth map
   |
   v
integration/pipeline.py: draws overlays, attaches a relative depth value to each tracked box
```

## Modules

| Module | File | Responsibility |
|--------|------|----------------|
| Detection | `src/detection/train.py`, `infer.py`, `export.py` | Train, run, and export YOLOv8 |
| Tracking | `src/tracking/bytetrack_tracker.py` | ByteTrack via Ultralytics |
| Lane detection | `src/lane_detection/lane_detector.py` | ROI mask, Canny, probabilistic Hough, left/right line fit |
| Depth | `src/depth_estimation/midas_depth.py` | MiDaS small model, relative depth only |
| Integration | `src/integration/pipeline.py` | Combines all modules per frame |

## Design notes

- MiDaS outputs relative inverse depth, not metric distance. Treat the per-box depth value as
  an ordering signal (nearer vs farther), not meters.
- Lane detection is classical and tuned for forward-facing, daylight, structured roads.
  It will degrade at night, in rain, and on unmarked roads.

## Pending: CARLA and ROS2 (not implemented)

Planned design:

1. A CARLA client spawns an ego vehicle with an RGB camera.
2. A ROS2 node subscribes to `/camera/image_raw`, runs `PerceptionPipeline.process_frame`,
   and publishes detections and the annotated image.
3. Evaluate closed-loop behavior in CARLA scenarios.

## Pending: TensorRT

`src/detection/export.py` can export to ONNX and TensorRT. Latency benchmarks on target
hardware are still to be collected and recorded in `docs/evaluation.md`.
