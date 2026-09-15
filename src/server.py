from no_emoji import strip_emojis_streams
strip_emojis_streams()

import base64
import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

import config

app = FastAPI(title="CarParts Segmenter API")

WEB_DIR = config.BASE_DIR / "web"

# Load the model on the first request so startup stays quick.
BEST_WEIGHTS = Path(config.TRAIN_PROJECT_DIR) / config.PROJECT_NAME / "weights" / "best.pt"
FALLBACK_WEIGHTS = "yolo26n-seg.pt"

_model: Optional[YOLO] = None
_is_custom_model = False


def get_model() -> YOLO:
    global _model, _is_custom_model
    if _model is None:
        if BEST_WEIGHTS.exists():
            _model = YOLO(str(BEST_WEIGHTS))
            _is_custom_model = True
        else:
            _model = YOLO(FALLBACK_WEIGHTS)
            _is_custom_model = False
    return _model


def class_names_list(model: YOLO) -> list[str]:
    names = model.names
    return list(names.values()) if isinstance(names, dict) else list(names)


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def run_and_render(image: Image.Image, conf: float, iou: float):
    model = get_model()
    img_array = np.array(image.convert("RGB"))

    start = time.time()
    results = model.predict(source=img_array, conf=conf, iou=iou, verbose=False)
    elapsed = time.time() - start
    result = results[0]

    # Keep text in the frontend so the compare slider does not cut through it.
    annotated = result.plot(labels=False, conf=False)
    annotated_rgb = annotated[..., ::-1]
    annotated_img = Image.fromarray(annotated_rgb)

    n_instances = 0 if result.boxes is None else len(result.boxes)
    detections = []
    if n_instances > 0:
        names = class_names_list(model)
        # Normalized coordinates let the frontend place labels at any size.
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = [round(float(v), 4) for v in box.xyxyn[0].tolist()]
            detections.append({
                "part": names[cls_id],
                "confidence": round(float(box.conf.item()), 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
        detections.sort(key=lambda d: d["confidence"], reverse=True)

    n_classes = len({d["part"] for d in detections})
    fps = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0

    return {
        "original_image": image_to_data_url(image.convert("RGB")),
        "annotated_image": image_to_data_url(annotated_img),
        "detections": detections,
        "stats": {
            "parts_detected": n_instances,
            "unique_classes": n_classes,
            "fps": fps,
            "elapsed_ms": round(elapsed * 1000, 1),
        },
    }


@app.get("/api/model-info")
def model_info():
    model = get_model()
    return {
        "is_custom": _is_custom_model,
        "classes": class_names_list(model),
        "default_conf": config.CONF_THRESHOLD,
    }


@app.get("/api/samples")
def list_samples():
    sample_dir = config.TEST_IMG_DIR
    files = sorted(
        p.name for p in list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.jpeg")) + list(sample_dir.glob("*.png"))
    )
    return {"samples": files}


@app.get("/api/samples/{filename}")
def get_sample(filename: str):
    path = config.TEST_IMG_DIR / filename
    if not path.exists() or path.parent != config.TEST_IMG_DIR:
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path)


@app.post("/api/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Form(None),
    conf: float = Form(config.CONF_THRESHOLD),
    iou: float = Form(0.5),
):
    if file is not None:
        raw = await file.read()
        try:
            image = Image.open(io.BytesIO(raw))
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read that image file")
    elif sample:
        path = config.TEST_IMG_DIR / sample
        if not path.exists() or path.parent != config.TEST_IMG_DIR:
            raise HTTPException(status_code=404, detail="Sample not found")
        image = Image.open(path)
    else:
        raise HTTPException(status_code=400, detail="Provide either a file upload or a sample name")

    try:
        payload = run_and_render(image, conf=conf, iou=iou)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return JSONResponse(payload)


# Mount the frontend after the API routes.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
