# Evaluation

Record measured results here. Nothing below is filled in until you measure it.

## Detection (YOLOv8, BDD100K val)

Source: `runs/detect/train/results.csv` from your Colab training run.

| Model | Epochs | Img size | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|--------|----------|---------|--------------|-----------|--------|
| YOLOv8 (fill in variant) | TBD | TBD | TBD | TBD | TBD | TBD |

Per-class AP: TBD (from `val` output).

## Tracking

Suggested metrics: HOTA, MOTA, IDF1 (requires a tracking-labeled split such as BDD100K MOT).

## Lane detection

Qualitative review on sample frames. Note failure cases (night, rain, worn markings).

## Latency

| Format | Hardware | ms/frame | FPS |
|--------|----------|----------|-----|
| PyTorch (.pt) | TBD | TBD | TBD |
| ONNX | TBD | TBD | TBD |
| TensorRT | TBD | TBD | TBD |
