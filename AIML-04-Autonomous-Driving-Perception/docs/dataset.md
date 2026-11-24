# Dataset: BDD100K

BDD100K is a large driving dataset from Berkeley DeepDrive. This project uses the
100K images with 2D box labels for detection.

Download from https://www.vis.xyz/bdd100k/ (registration required). Respect the dataset license.

## Classes (10)

pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign

## Converting labels to YOLO format

BDD100K ships labels as JSON with `box2d` entries (`x1, y1, x2, y2`) on 1280x720 images.
For each box, write a line to `labels/<split>/<image>.txt`:

```
<class_id> <x_center/1280> <y_center/720> <width/1280> <height/720>
```

Expected layout:

```
data/bdd100k_yolo/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

`notebooks/02_dataset_exploration.ipynb` contains the conversion code and class-distribution EDA.

## EDA checklist

- Class frequency (cars dominate; rider, train, motorcycle are rare)
- Box size distribution (many small traffic lights and signs)
- Weather, scene, and time-of-day attribute breakdown
- Objects per image
