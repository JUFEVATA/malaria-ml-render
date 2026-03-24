from datetime import datetime
from pathlib import Path
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow.keras.models import load_model

APP_TITLE = "Malaria Prediction API"
MODEL_NAME = "lenet.keras"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / MODEL_NAME

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
INPUT_SHAPE = None
IMG_HEIGHT = 224
IMG_WIDTH = 224

metrics_data = {
    "total_predictions": 0,
    "parasitized_count": 0,
    "uninfected_count": 0,
    "scores": [],
    "last_prediction": None,
    "model_version": MODEL_NAME,
}


def load_trained_model() -> None:
    global model, INPUT_SHAPE, IMG_HEIGHT, IMG_WIDTH

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    INPUT_SHAPE = model.input_shape

    if len(INPUT_SHAPE) != 4:
        raise ValueError(f"Forma de entrada no válida del modelo: {INPUT_SHAPE}")

    IMG_HEIGHT = INPUT_SHAPE[1]
    IMG_WIDTH = INPUT_SHAPE[2]

    if IMG_HEIGHT is None or IMG_WIDTH is None:
        raise ValueError(
            f"No se pudo determinar el tamaño de entrada del modelo: {INPUT_SHAPE}"
        )


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as exc:
        raise ValueError(f"Error al preprocesar la imagen: {exc}") from exc


def validate_uploaded_file(file: UploadFile) -> None:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen válida",
        )


def interpret_prediction(pred: np.ndarray) -> tuple[str, float]:
    if len(pred.shape) == 2 and pred.shape[1] == 1:
        prob = float(pred[0][0])

        # En este modelo:
        # 0 -> Parasitized
        # 1 -> Uninfected
        if prob < 0.5:
            return "Parasitized", round((1 - prob), 4)
        return "Uninfected", round(prob, 4)

    if len(pred.shape) == 2 and pred.shape[1] == 2:
        class_idx = int(np.argmax(pred[0]))
        labels = ["Parasitized", "Uninfected"]
        score = float(pred[0][class_idx])
        return labels[class_idx], round(score, 4)

    raise ValueError(f"Salida del modelo no esperada. prediction_shape={pred.shape}")


def update_metrics(label: str, score_percent: float) -> None:
    metrics_data["total_predictions"] += 1
    metrics_data["scores"].append(score_percent)

    if label == "Parasitized":
        metrics_data["parasitized_count"] += 1
    else:
        metrics_data["uninfected_count"] += 1

    metrics_data["last_prediction"] = {
        "label": label,
        "score": score_percent,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_average_score() -> float:
    if not metrics_data["scores"]:
        return 0.0
    return round(sum(metrics_data["scores"]) / len(metrics_data["scores"]), 2)


@app.on_event("startup")
def startup_event() -> None:
    load_trained_model()


@app.get("/")
def home():
    return {
        "message": "API de malaria activa",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_version": metrics_data["model_version"],
        "input_shape": str(INPUT_SHAPE) if INPUT_SHAPE else None,
    }


@app.get("/metrics")
def metrics():
    return {
        "total_predictions": metrics_data["total_predictions"],
        "parasitized_count": metrics_data["parasitized_count"],
        "uninfected_count": metrics_data["uninfected_count"],
        "average_score": get_average_score(),
        "model_version": metrics_data["model_version"],
    }


@app.get("/last_prediction")
def last_prediction():
    if metrics_data["last_prediction"] is None:
        return {"message": "Aún no se ha realizado ninguna predicción"}
    return metrics_data["last_prediction"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado")

    validate_uploaded_file(file)

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo está vacío")

        image = preprocess_image(image_bytes)
        prediction = model.predict(image, verbose=0)

        try:
            label, score = interpret_prediction(prediction)
        except ValueError:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Salida del modelo no esperada",
                    "prediction_shape": str(prediction.shape),
                    "raw_prediction": prediction.tolist(),
                },
            )

        score_percent = round(score * 100, 2)
        update_metrics(label, score_percent)

        return {
            "label": label,
            "score": score_percent,
            "input_shape_model": str(INPUT_SHAPE),
            "prediction_shape": str(prediction.shape),
            "filename": file.filename,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en la predicción: {exc}",
        ) from exc