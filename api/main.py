import io, json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from ultralytics import YOLO

MODELS = Path(__file__).resolve().parent.parent / "models"
state = {}

app = FastAPI(title="Brain Tumor API")


@app.on_event("startup")
def load_models():
    # Models are loaded once at startup
    state["cls"] = tf.keras.models.load_model(MODELS / "inceptionv3_ft.keras")
    state["names"] = json.load(open(MODELS / "class_names.json"))
    state["det"] = YOLO(str(MODELS / "best.pt"))


def read_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify(file: UploadFile = File(...)):
    img = read_image(file).resize((299, 299))
    x = preprocess_input(np.expand_dims(np.asarray(img, dtype=np.float32), 0))
    probs = state["cls"].predict(x, verbose=0)[0]
    best = int(probs.argmax())
    return {
        "class": state["names"][best],
        "confidence": float(probs[best]),
        "scores": {n: float(p) for n, p in zip(state["names"], probs)},
    }


@app.post("/detect")
def detect(file: UploadFile = File(...)):
    img = read_image(file)
    res = state["det"].predict(img, conf=0.25, imgsz=640, verbose=False)[0]
    boxes = [
        {
            "class": res.names[int(b.cls)],
            "confidence": float(b.conf),
            "box": [float(v) for v in b.xyxy[0]],   # x1, y1, x2, y2 in pixels
        }
        for b in res.boxes
    ]
    return {"image_size": list(img.size), "detections": boxes}