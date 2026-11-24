# AIML-04 – Autonomous Driving Perception Stack

A modular perception pipeline for driving scenes built on the **BDD100K** dataset:
object detection (YOLOv8), multi-object tracking (ByteTrack), classical lane detection (OpenCV),
monocular depth estimation (MiDaS), and an integration layer that fuses them per frame.

## Project status

| # | Module | Status |
|---|--------|--------|
| 1 | BDD100K preparation and EDA | Done (see `notebooks/02_dataset_exploration.ipynb`) |
| 2 | YOLOv8 training and evaluation | Done in Colab (see `notebooks/03_yolo_training.ipynb`) |
| 3 | Detection + ByteTrack tracking | Code provided, run on your own videos to validate |
| 4 | OpenCV lane detection | Code provided with unit tests |
| 5 | MiDaS depth estimation | Code provided, run on your own frames to validate |
| 6 | CARLA / ROS2 integration | Pending (design in `docs/architecture.md`) |
| 7 | ONNX / TensorRT optimization | Export script provided, benchmarking pending |
| 8 | Weights & Biases tracking | Optional hook provided in `src/detection/train.py` |

"Code provided" means the module is implemented but its accuracy and speed numbers have not
been measured yet. Record real numbers in `docs/evaluation.md` as you collect them.

## Repository layout

```
configs/        dataset config (data.yaml)
notebooks/      Colab-friendly notebooks (setup, EDA, training, tracking)
src/
  detection/        training, inference, export
  tracking/         ByteTrack wrapper
  lane_detection/   Canny + Hough lane detector
  depth_estimation/ MiDaS wrapper
  integration/      per-frame fusion pipeline
weights/        trained weights (git-ignored, keep best.pt in Google Drive)
results/        plots and metrics
tests/          unit tests
docs/           architecture, dataset, evaluation notes
```

## Quick start

```bash
git clone https://github.com/<your-username>/AIML-04-Autonomous-Driving-Perception.git
cd AIML-04-Autonomous-Driving-Perception
pip install -r requirements.txt
```

Train (after preparing the dataset, see `docs/dataset.md`):

```bash
python -m src.detection.train --data configs/data.yaml --epochs 50 --imgsz 640
```

Run the full pipeline on a video:

```bash
python -m src.integration.pipeline --video input.mp4 --weights weights/best.pt --out results/output.mp4
```

Add `--depth` to enable MiDaS depth (slower).

## Results

Fill these in from your Colab run (`runs/detect/train/results.csv`):

| Metric | Value |
|--------|-------|
| mAP@0.5 | TBD |
| mAP@0.5:0.95 | TBD |
| Precision | TBD |
| Recall | TBD |

## Notes

- The dataset and `best.pt` are not stored in this repository. Keep your Colab notebook and
  trained weights in Google Drive as the source of truth.
- BDD100K is licensed separately, see https://www.vis.xyz/bdd100k/ for terms.

## License

MIT (code only).
