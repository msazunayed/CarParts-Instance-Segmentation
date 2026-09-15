# CarParts Segmenter

Instance segmentation pipeline that detects and masks individual car parts
(bumpers, doors, headlights, mirrors, hood, trunk, glass, wheels, etc.) from
a single image or live webcam feed, using YOLO26-seg and the Ultralytics
**Carparts-Seg** dataset.

## Why this project

Pixel-level part segmentation (not just bounding boxes) supports real
use cases:
- **Auto repair / insurance claims** — identify exactly which part is
  damaged and its precise extent
- **Automotive manufacturing QC** — flag defects or misalignment on
  specific components
- **E-commerce cataloging** — auto-crop and tag individual parts from
  listing photos
- **Autonomous-vehicle perception** — fine-grained understanding of
  vehicle structure

## Dataset

**Carparts-Seg** (Ultralytics official, CC-BY 4.0 sourced from Gianmarco
Russo): 3,833 images, 20,374 polygon instances across 23 car-part classes,
pre-split into train (3,156) / val (401) / test (276). Downloads
automatically (~133 MB) the first time training runs — no manual setup.

## Project structure

```
carparts_segmenter/
├── src/
│   ├── config.py          # central config: dataset, weights, hyperparameters
│   ├── device.py          # GPU/CPU selection with free-memory checks
│   ├── trainer.py         # training loop with automatic CPU fallback on CUDA OOM
│   ├── evaluator.py        # loads best weights, runs validation, reports box + mask mAP
│   ├── exporter.py         # exports trained model to ONNX
│   ├── main.py             # orchestrates: device -> train -> validate -> export
│   ├── predict_images.py   # batch inference on a folder of test images
│   ├── detect_webcam.py     # live webcam segmentation demo with FPS overlay
│   └── server.py            # FastAPI backend for the web UI
├── web/                  # HTML/CSS/JS frontend served by server.py
│   ├── index.html
│   ├── app.js
│   └── style.css
├── requirements.txt
├── data/
│   └── test_images/    # drop your own images here for predict_images.py
└── models/              # pretrained yolo26n-seg.pt lands here on first run
```

All commands below are run from the **project root** (`carparts_segmenter/`),
not from inside `src/`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train, validate, and export in one command:

```bash
python src/main.py
```

This will:
1. Pick GPU if available (falls back to CPU automatically, including on
   CUDA out-of-memory mid-training)
2. Train `yolo26n-seg` on Carparts-Seg for 60 epochs (early stopping via
   `patience=15`)
3. Load the best checkpoint and report **box** and **mask** mAP50 / mAP50-95
4. Export the trained model to ONNX for deployment

Run inference on your own images:

```bash
python src/predict_images.py
```

Run live webcam segmentation:

```bash
python src/detect_webcam.py
```

Launch the web UI (upload images, adjust confidence/IoU thresholds, view
per-part detections, download annotated results):

```bash
uvicorn --app-dir src server:app --reload --port 8000
```

Then open **http://localhost:8000**. This runs a FastAPI backend
(`src/server.py`) that serves the custom frontend in `web/` and exposes
a small JSON/multipart API. It automatically uses your trained `best.pt`
once `main.py` has run; otherwise it falls back to the stock
`yolo26n-seg.pt` weights.

## Results

After training, fill in your actual numbers here for your resume/README:

| Metric | Box | Mask |
|---|---|---|
| mAP50 | TBD | TBD |
| mAP50-95 | TBD | TBD |

## Suggested resume bullet

> Built an end-to-end YOLO26 instance segmentation pipeline to detect and
> mask 23 individual car-part classes from images and live video, achieving
> [X]% mask mAP50 on the Carparts-Seg dataset; exported to ONNX for
> deployment and benchmarked inference at [Y] FPS.
