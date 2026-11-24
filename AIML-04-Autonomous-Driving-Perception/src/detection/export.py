"""Export trained weights to ONNX or TensorRT.

TensorRT export needs an NVIDIA GPU with TensorRT installed.
"""
import argparse


def main():
    p = argparse.ArgumentParser(description="Export YOLOv8 weights")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--format", default="onnx", choices=["onnx", "engine"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 (GPU only)")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print("Exported to:", out)


if __name__ == "__main__":
    main()
