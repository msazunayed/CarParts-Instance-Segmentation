from no_emoji import strip_emojis_streams
strip_emojis_streams()

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

import config

MODEL_PATH = Path(config.TRAIN_PROJECT_DIR) / config.PROJECT_NAME / "weights" / "best.pt"
CONF_THRESHOLD = config.CONF_THRESHOLD
CAM_INDEX = 0


def main():
    model_path = str(MODEL_PATH) if MODEL_PATH.exists() else "yolo26n-seg.pt"
    if MODEL_PATH.exists():
        print(f"Using trained weights: {MODEL_PATH}")
    else:
        print(f"Trained weights not found at {MODEL_PATH}. Falling back to {model_path}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    prev_time = 0.0
    print("Press 'q' to quit, 's' to save current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        # Ultralytics draws the masks and labels for us.
        annotated_frame = results[0].plot()

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("CarParts Segmenter - Live", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
