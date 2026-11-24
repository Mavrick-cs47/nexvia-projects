"""Run a trained YOLOv8 detector on an image, folder, or video."""
import argparse


def main():
    p = argparse.ArgumentParser(description="YOLOv8 inference")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--source", required=True, help="image, folder, or video path")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, save=True)


if __name__ == "__main__":
    main()
