# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate venv first (Windows): `venv\Scripts\activate`

- Train: `python 01_train_model.py` (~30-45 min on RTX 4050, downloads ~800MB Oxford-IIIT Pet dataset first run)
- Evaluate / generate thesis figures: `python 02_evaluate_model.py` (requires `outputs/models/final_model.keras` from training)
- Launch Gradio UI (quick demo): `python 03_app.py` → `http://127.0.0.1:7860`
- Launch FastAPI + editorial frontend (preferred UI): `python 04_server.py` → `http://127.0.0.1:8000`
- Install deps: `pip install -r requirements.txt`
- GPU check: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

No test suite, no linter configured.

## Architecture

Four-script pipeline (train → evaluate → two UIs), communicates via files in `outputs/`. No shared module — config constants (`IMG_SIZE=224`, `BATCH_SIZE=16`, `NUM_CLASSES=37`) are duplicated across scripts. Keep them in sync when editing.

**`01_train_model.py`** — EfficientNetV2B0 transfer learning, two-phase:
1. Phase 1: base frozen, train head only at LR=1e-3 (10 epochs)
2. Phase 2: unfreeze base, fine-tune entire model at LR=1e-5 (10 epochs). Low LR critical — higher destroys pretrained weights.

Head = GAP → Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(37, softmax). Augmentation (flip/rotate/zoom/contrast) baked into model graph via `data_augmentation` Sequential, runs only when `training=True`.

GPU memory growth enabled at script start — required for 6GB laptop GPUs (RTX 4050) to avoid VRAM OOM. If OOM still hits, lower `BATCH_SIZE` (16 → 8 → 4).

**`02_evaluate_model.py`** — loads `outputs/models/final_model.keras` + `outputs/class_names.txt`, regenerates test split via `tfds.load`, produces confusion matrix, classification report, per-class accuracy chart, sample predictions grid.

**`04_server.py` + `web/`** — primary UI. FastAPI backend loads model once, exposes `POST /api/predict` (base64 image → top-5 JSON with species + low-confidence flag) and `GET /api/breeds` (catalog grouped cat/dog). Static frontend under `web/` (`index.html`, `style.css`, `app.js`) is hand-built editorial design: Fraunces display + Newsreader body + JetBrains Mono numerals, asymmetric grid, paper grain via inline SVG noise, animated probability bars. Webcam mode uses `getUserMedia`, captures via canvas at `STREAM_INTERVAL_MS = 280` (`web/app.js`), downscales to 320px before POST. Frame mirroring is reversed before sending so the model sees the natural orientation.

**`03_app.py`** — legacy Gradio UI, kept for quick local demos. Loads `final_model.keras` + `class_names.txt` once at startup. Two tabs sharing one `predict(img)` fn: (1) Snapshot — upload / clipboard / webcam snap with Classify button; (2) Live camera — `gr.Image(streaming=True)` bound via `.stream()` with `stream_every=0.25`. Outputs top-5 via `gr.Label`, derives cat/dog from class-name casing (capitalized = cat), warns when top-1 prob < `LOW_CONF_THRESHOLD = 0.50`. Custom theme + CSS, supported-breeds accordion built from `class_names.txt`. `demo.queue(max_size=1)` drops backed-up frames so live feed shows latest.

## Outputs contract

`01_*` writes: `outputs/models/{best_phase1,best_final,final_model}.keras`, `outputs/class_names.txt`, `outputs/figures/training_curves.png`, `outputs/results_summary.txt`.

`02_*` reads `final_model.keras` + `class_names.txt`, writes: `outputs/figures/{confusion_matrix,sample_predictions,per_class_accuracy}.png`, `outputs/classification_report.txt`.

## Constraints

- Python 3.10/3.11 only. TF 2.15 incompatible with 3.12+.
- CUDA 11.8 + cuDNN 8.6 required for GPU.
- EfficientNetV2B0 expects 224×224 — don't change `IMG_SIZE` without verifying input compat.
- Images NOT normalized to [0,1] — kept as float32 [0,255]. EfficientNetV2 preprocessing is built into the Keras app.
