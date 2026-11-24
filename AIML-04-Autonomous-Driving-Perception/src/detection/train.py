"""Train YOLOv8 on the BDD100K YOLO-format dataset."""
import argparse


def main():
    p = argparse.ArgumentParser(description="Train YOLOv8 on BDD100K")
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="starting checkpoint, e.g. yolov8s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="train")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    args = p.parse_args()

    from ultralytics import YOLO, settings

    if args.wandb:
        # Ultralytics has a built-in W&B callback, enabled through its settings.
        settings.update({"wandb": True})

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    metrics = model.val()
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
