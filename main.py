import base64
import os
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="YOLO Food Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_SIZE = 416  # Try 416 first; use 320 for more speed, 640 for best accuracy
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model_path = os.path.join(os.getcwd(), "best.pt")
model = YOLO(model_path) if os.path.exists(model_path) else None

if model:
    # One-time warm-up: prevents the first real request from being unusually slow.
    model.predict(
        np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8),
        imgsz=MODEL_SIZE,
        device=DEVICE,
        verbose=False,
    )

class ImageRequest(BaseModel):
    base64_image: str

@app.post("/predict")
def predict_ingredients(request: ImageRequest):  # regular def: FastAPI runs it in a worker thread
    if model is None:
        raise HTTPException(500, "Model best.pt is not loaded.")

    try:
        start = time.perf_counter()

        image_data = base64.b64decode(request.base64_image)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")

        height, width = img.shape[:2]

        results = model.predict(
            source=img,
            imgsz=MODEL_SIZE,
            conf=0.25,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]
        predictions = []

        for box in result.boxes:
            x, y, w, h = box.xywh[0].tolist()
            class_id = int(box.cls[0])

            predictions.append({
                "class": result.names[class_id],
                "confidence": round(float(box.conf[0]), 3),
                "x": round(x, 1),
                "y": round(y, 1),
                "width": round(w, 1),
                "height": round(h, 1),
            })

        return {
            "predictions": predictions,
            "image": {"width": width, "height": height},
            "time": round(time.perf_counter() - start, 3),
        }

    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)  # no reload=True