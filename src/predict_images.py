from no_emoji import strip_emojis_streams
strip_emojis_streams()

from pathlib import Path
from ultralytics import YOLO

import config

MODEL_PATH = Path(config.TRAIN_PROJECT_DIR) / config.PROJECT_NAME / "weights" / "best.pt"


def main():
    model_path = str(MODEL_PATH) if MODEL_PATH.exists() else "yolo26n-seg.pt"
    if MODEL_PATH.exists():
        print(f"Using trained weights: {MODEL_PATH}")
    else:
        print(f"Trained weights not found at {MODEL_PATH}. Falling back to {model_path}")

    model = YOLO(model_path)

    results = model.predict(
        source=config.TEST_SOURCE,
        conf=config.CONF_THRESHOLD,
        save=True,
        project=config.PREDICT_PROJECT_DIR,
        name=config.PROJECT_NAME,
        exist_ok=True,
    )
    print(f"\nSaved {len(results)} annotated image(s) to "
          f"{config.PREDICT_PROJECT_DIR}/{config.PROJECT_NAME}")


if __name__ == "__main__":
    main()
