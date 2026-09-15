# Loads the trained model and runs it against the validation set.



from pathlib import Path
from ultralytics import YOLO
import config


def load_best_model(fallback_model: YOLO) -> YOLO:
    best = Path(config.TRAIN_PROJECT_DIR) / config.PROJECT_NAME / "weights" / "best.pt"
    if best.exists():
        return YOLO(str(best))
    print(f"Trained weights not found at {best}. Using last in-memory model instead.")
    return fallback_model


def validate_model(model: YOLO, device: str):
    metrics = model.val(data=config.DATASET, device=device)
    print("\n--- Box metrics ---")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("\n--- Mask metrics ---")
    print(f"mAP50:    {metrics.seg.map50:.4f}")
    print(f"mAP50-95: {metrics.seg.map:.4f}")
    return metrics
