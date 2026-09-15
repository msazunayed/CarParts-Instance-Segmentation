# Settings shared by the car parts segmentation scripts.

from pathlib import Path

# The project root contains the data and model folders.
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TEST_IMG_DIR = DATA_DIR / "test_images"

# Make sure the working folders exist before anything uses them.
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Ultralytics downloads this dataset the first time training runs.
DATASET = "carparts-seg.yaml"

# The -seg weights return masks as well as bounding boxes.
WEIGHTS = str(MODEL_DIR / "yolo26n-seg.pt")

EPOCHS = 60
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 15

PROJECT_NAME = "CarParts_Segmenter"
RUNS_DIR = BASE_DIR / "runs"
TRAIN_PROJECT_DIR = str(RUNS_DIR / "train")
PREDICT_PROJECT_DIR = str(RUNS_DIR / "predict")
TEST_SOURCE = str(TEST_IMG_DIR)

CONF_THRESHOLD = 0.25
MIN_FREE_GB = 1.0
