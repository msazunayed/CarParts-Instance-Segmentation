# Train the model, retrying on CPU if we hit a CUDA OOM.

from ultralytics import YOLO
import config


def _train_once(model: YOLO, device: str):
    # Run one training attempt; retry handling belongs to the caller.
    return model.train(
        data=config.DATASET,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        device=device,
        batch=config.BATCH_SIZE,
        project=config.TRAIN_PROJECT_DIR,
        name=config.PROJECT_NAME,
        patience=config.PATIENCE,
        amp=False,
        exist_ok=True,
    )


def train_model(device: str) -> tuple[YOLO, str]:
    # Retry on the CPU if CUDA runs out of memory.
    model = YOLO(config.WEIGHTS)

    try:
        _train_once(model, device)
        return model, device
    except RuntimeError as e:
        print(f"\nCUDA out of memory during training: {e}")
        print("Retrying on CPU...\n")
        device = "cpu"
        model = YOLO(config.WEIGHTS)
        _train_once(model, device)
        return model, device
