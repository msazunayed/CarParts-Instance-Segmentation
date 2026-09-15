# Export the trained model in the requested format.

from ultralytics import YOLO


def export_model(model: YOLO, fmt: str = "onnx"):
    model.export(format=fmt)
