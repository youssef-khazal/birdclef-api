"""FastAPI service: upload a recording, get the most likely species.

    uvicorn birdclef_api.app:app --host 0.0.0.0 --port 7860

Endpoints:
  GET  /           demo page (upload a file or try an example)
  GET  /health     liveness + model info
  GET  /species    the 206 species the model knows
  POST /predict    multipart file upload -> top-k species, per-segment predictions, timing
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from birdclef_api import __version__
from birdclef_api.model import SpeciesClassifier, resource_dir
from birdclef_api.preprocessing import SAMPLE_RATE, load_audio

MAX_UPLOAD_MB = 15
MAX_SECONDS = 120  # longer recordings are truncated (keeps the free CPU responsive)
STATIC = Path(__file__).parent / "static"
EXAMPLES = resource_dir("examples")

app = FastAPI(
    title="BirdCLEF species classifier",
    description="Identify 206 species (birds, amphibians, insects, mammals) from a field recording.",
    version=__version__,
)
classifier = SpeciesClassifier()
classifier.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm-up: librosa/numba compile on first call


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": classifier.model_file, "classes": classifier.n_classes, "version": __version__}


@app.get("/species")
def species() -> list[dict]:
    return classifier.labels


@app.get("/examples/{name}", include_in_schema=False)
def example(name: str) -> FileResponse:
    path = (EXAMPLES / name).resolve()
    if path.parent != EXAMPLES.resolve() or path.suffix != ".ogg" or not path.exists():
        raise HTTPException(404, "Unknown example")
    return FileResponse(path, media_type="audio/ogg")


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Audio file: wav, flac, ogg, mp3 (m4a with ffmpeg)"),
                  top_k: int = Query(5, ge=1, le=20)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    t0 = time.perf_counter()
    try:
        waveform = load_audio(data)
    except Exception:
        raise HTTPException(415, "Could not decode this audio file") from None
    if len(waveform) < SAMPLE_RATE // 2:
        raise HTTPException(400, "Recording shorter than 0.5 s")
    truncated = len(waveform) > MAX_SECONDS * SAMPLE_RATE
    waveform = waveform[: MAX_SECONDS * SAMPLE_RATE]
    t1 = time.perf_counter()
    result = classifier.predict(waveform, top_k=top_k)
    t2 = time.perf_counter()
    result.update({
        "filename": file.filename, "truncated_to_seconds": MAX_SECONDS if truncated else None,
        "timing_ms": {"decode": round((t1 - t0) * 1000, 1), "inference": round((t2 - t1) * 1000, 1)},
    })
    return result
