"""
Pet Breed Classifier - FastAPI backend + static editorial frontend.
Run: python 04_server.py
Then open: http://127.0.0.1:8000
"""
import base64
import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- GPU memory growth ----
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# ---- Config ----
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / 'outputs' / 'models' / 'final_model.keras'
CLASS_NAMES_PATH = ROOT / 'outputs' / 'class_names.txt'
WEB_DIR = ROOT / 'web'
IMG_SIZE = 224
LOW_CONF_THRESHOLD = 0.50
TOP_K = 5

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH} not found. Run 01_train_model.py first.")

print("Loading model...")
model = keras.models.load_model(str(MODEL_PATH))

with open(CLASS_NAMES_PATH, 'r') as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]


def pretty(name: str) -> str:
    return name.replace('_', ' ').title()


def species_of(name: str) -> str:
    # Oxford-IIIT convention: capitalized = cat, lowercase = dog
    return "cat" if name[0].isupper() else "dog"


CAT_BREEDS = sorted([c for c in CLASS_NAMES if species_of(c) == 'cat'])
DOG_BREEDS = sorted([c for c in CLASS_NAMES if species_of(c) == 'dog'])

print(f"Loaded {len(CLASS_NAMES)} classes "
      f"({len(CAT_BREEDS)} cats, {len(DOG_BREEDS)} dogs)")


# ---- App ----
app = FastAPI(title="Pet Breed Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # data URL or raw base64


class Prediction(BaseModel):
    rank: int
    name: str
    pretty: str
    species: str
    prob: float


class PredictResponse(BaseModel):
    top: list[Prediction]
    species: str
    low_confidence: bool
    threshold: float


def decode_image(data: str) -> np.ndarray:
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return np.asarray(img)


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    arr = decode_image(req.image)
    x = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE))
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, 0)
    probs = model.predict(x, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:TOP_K]

    top = [
        Prediction(
            rank=i + 1,
            name=CLASS_NAMES[j],
            pretty=pretty(CLASS_NAMES[j]),
            species=species_of(CLASS_NAMES[j]),
            prob=float(probs[j]),
        )
        for i, j in enumerate(idx)
    ]
    top1 = top[0]
    return PredictResponse(
        top=top,
        species=top1.species,
        low_confidence=top1.prob < LOW_CONF_THRESHOLD,
        threshold=LOW_CONF_THRESHOLD,
    )


@app.get("/api/breeds")
def breeds():
    return {
        "cats": [{"name": b, "pretty": pretty(b)} for b in CAT_BREEDS],
        "dogs": [{"name": b, "pretty": pretty(b)} for b in DOG_BREEDS],
        "total": len(CLASS_NAMES),
        "threshold": LOW_CONF_THRESHOLD,
    }


# Static frontend
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
