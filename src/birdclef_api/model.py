"""ONNX Runtime wrapper: audio in, ranked species out."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from birdclef_api.preprocessing import SAMPLE_RATE, SEGMENT_SECONDS, audio_to_batch

def resource_dir(name: str) -> Path:
    """Find ``models/`` or ``examples/``: env var (MODEL_DIR / EXAMPLES_DIR), then the repo, then the working dir."""
    env = os.environ.get({"models": "MODEL_DIR", "examples": "EXAMPLES_DIR"}[name])
    candidates = [Path(env)] if env else []
    candidates += [Path(__file__).resolve().parents[2] / name, Path.cwd() / name]
    return next((c for c in candidates if c.is_dir()), candidates[-1])


DEFAULT_MODEL_DIR = resource_dir("models")


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Prediction:
    code: str
    common_name: str
    scientific_name: str
    group: str
    probability: float


class SpeciesClassifier:
    """Classifies every 5 s segment of a recording and averages the probabilities (as in training/eval)."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, model_file: str = "efficientnet_b0.onnx", threads: int | None = None):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / model_file), opts, providers=["CPUExecutionProvider"])
        self.labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.model_file = model_file

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def segment_probabilities(self, waveform: np.ndarray, batch_size: int = 16) -> np.ndarray:
        batch = audio_to_batch(waveform)
        logits = np.concatenate(
            [self.session.run(None, {"spectrogram": batch[i : i + batch_size]})[0] for i in range(0, len(batch), batch_size)]
        )
        return softmax(logits)

    def _to_predictions(self, probs: np.ndarray, top_k: int) -> list[Prediction]:
        best = np.argsort(-probs)[:top_k]
        return [Prediction(probability=float(probs[i]), **self.labels[i]) for i in best]

    def predict(self, waveform: np.ndarray, top_k: int = 5) -> dict:
        seg_probs = self.segment_probabilities(waveform)
        return {
            "duration_seconds": round(len(waveform) / SAMPLE_RATE, 2),
            "segments": len(seg_probs),
            "top_k": [p.__dict__ for p in self._to_predictions(seg_probs.mean(axis=0), top_k)],
            "per_segment": [
                {"start_seconds": k * SEGMENT_SECONDS, **self._to_predictions(p, 1)[0].__dict__}
                for k, p in enumerate(seg_probs)
            ],
        }
