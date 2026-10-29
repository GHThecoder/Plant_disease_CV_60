"""Local EfficientNet-B2 ONNX inference with the training eval preprocess."""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from labels_tr import TR_LABELS

DATASET_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = DATASET_ROOT / "runs" / "exported" / "efficientnet_b2"
META_PATH = EXPORT_DIR / "efficientnet_b2_export_meta.json"
ONNX_PATH = EXPORT_DIR / "efficientnet_b2.onnx"

# script1.build_transforms(train=False): shorter edge 256, then 224 center crop.
RESIZE_SHORT_EDGE = 256
TOP_K = 4

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
IMAGENET_MEAN = tuple(float(x) for x in meta["imagenet_mean"])
IMAGENET_STD = tuple(float(x) for x in meta["imagenet_std"])
IMG_SIZE = int(meta["img_size"])
CLASS_NAMES: list[str] = list(meta["class_names"])
if len(TR_LABELS) != len(CLASS_NAMES):
    raise RuntimeError(
        f"Türkçe sınıf listesi {len(TR_LABELS)}, model {len(CLASS_NAMES)}"
    )

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(RESIZE_SHORT_EDGE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

session = None
input_name = "input"
provider_name = ""
fixed_batch: int | None = None


def class_info(index: int) -> dict:
    label = TR_LABELS[index]
    crop, condition = label.split(" - ", 1)
    folded = condition.casefold()
    healthy = folded == "sağlıklı"
    if healthy:
        kind = "sağlıklı"
    elif "zararlı" in folded:
        kind = "zararlı"
    else:
        kind = "hastalık"
    return {
        "class_index": index,
        "class_name": CLASS_NAMES[index],
        "label": label,
        "crop": crop,
        "disease": condition,
        "healthy": healthy,
        "kind": kind,
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max()
    exp = np.exp(z)
    return exp / exp.sum()


def load_session():
    import onnxruntime as ort

    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

    available = ort.get_available_providers()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    errors: list[str] = []
    for provider in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if provider not in available:
            continue
        try:
            sess = ort.InferenceSession(
                str(ONNX_PATH), sess_options=options, providers=[provider]
            )
            used = sess.get_providers()[0]
            return sess, used
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}: {exc}")
    raise RuntimeError(
        "ONNX Runtime could not start. "
        f"Available providers={available}. Details: {errors}"
    )


def preprocess(data: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Bu dosya görsel olarak okunamadı") from exc
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).contiguous()
    return tensor.numpy().astype(np.float32, copy=False)


def run_onnx(batch: np.ndarray) -> np.ndarray:
    assert session is not None
    if fixed_batch is None or batch.shape[0] == fixed_batch:
        return session.run(None, {input_name: batch})[0]
    if batch.shape[0] > fixed_batch:
        raise RuntimeError("Internal batch larger than the exported graph")
    pad = np.zeros((fixed_batch, *batch.shape[1:]), dtype=batch.dtype)
    pad[: batch.shape[0]] = batch
    out = session.run(None, {input_name: pad})[0]
    return out[: batch.shape[0]]


def predict_image(filename: str, data: bytes) -> dict:
    logits = run_onnx(preprocess(data))[0]
    if logits.shape[0] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model çıktısı {logits.shape[0]} sınıf, beklenen {len(CLASS_NAMES)}"
        )
    probs = softmax(logits)
    order = np.argsort(probs)[::-1][:TOP_K]
    top4 = []
    for rank, index in enumerate(order, start=1):
        hit = class_info(int(index))
        hit["rank"] = rank
        hit["probability"] = float(probs[index])
        top4.append(hit)
    return {"filename": filename, "error": None, "top4": top4}


def preprocessing_payload() -> dict:
    return {
        "resize_short_edge": RESIZE_SHORT_EDGE,
        "center_crop": IMG_SIZE,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "layout": meta.get("input_layout", "NCHW"),
        "dtype": meta.get("input_dtype", "float32"),
        "temperature_baked": bool(meta.get("temperature_baked", True)),
        "temperature": float(meta.get("temperature", 1.0)),
        "note": "Her görsel RGB, kısa kenar 256, merkez 224 kırpım, ardından (x - mean) / std.",
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global session, input_name, provider_name, fixed_batch
    session, provider_name = load_session()
    inp = session.get_inputs()[0]
    input_name = inp.name
    shape = inp.shape
    fixed_batch = None
    if shape and isinstance(shape[0], int) and shape[0] > 0:
        fixed_batch = int(shape[0])
    yield
    session = None


app = FastAPI(title="Bitki tanı", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": session is not None, "provider": provider_name}


@app.get("/api/meta")
def model_meta() -> dict:
    return {
        "model": meta.get("backbone", "efficientnet_b2"),
        "num_classes": len(CLASS_NAMES),
        "provider": provider_name,
        "img_size": IMG_SIZE,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "temperature_baked": bool(meta.get("temperature_baked", True)),
        "temperature": float(meta.get("temperature", 1.0)),
        "preprocessing": preprocessing_payload(),
    }


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)) -> dict:
    if session is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    if not files:
        raise HTTPException(status_code=400, detail="En az bir görsel gerekli")

    results = []
    for upload in files:
        filename = upload.filename or "gorsel"
        data = await upload.read()
        if not data:
            results.append(
                {"filename": filename, "error": "Dosya boş", "top4": []}
            )
            continue
        try:
            results.append(predict_image(filename, data))
        except ValueError as exc:
            results.append({"filename": filename, "error": str(exc), "top4": []})
    return {
        "model": meta.get("backbone", "efficientnet_b2"),
        "provider": provider_name,
        "preprocessing": preprocessing_payload(),
        "results": results,
    }
