import json
from pathlib import Path

import numpy as np
import pytest

from birdclef_api.model import SpeciesClassifier, softmax

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(axis=1), 1) and np.isfinite(p).all()


def test_labels_cover_every_output():
    labels = json.loads((MODELS / "labels.json").read_text(encoding="utf-8"))
    clf = SpeciesClassifier()
    assert clf.session.get_outputs()[0].shape[1] == len(labels) == 206


def test_onnx_matches_pytorch_checkpoint():
    """Only runs when torch and the original checkpoint are available (not in the Docker image)."""
    torch = pytest.importorskip("torch")
    ckpt = Path(__file__).resolve().parents[2] / "birdclef-2025" / "runs" / "efficientnet_b0_recording_seed42" / "best.pth"
    if not ckpt.exists():
        pytest.skip("original checkpoint not found")
    from scripts.export_onnx import build_efficientnet_b0

    model = build_efficientnet_b0(206)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    from birdclef_api.preprocessing import audio_to_batch, load_audio

    x = audio_to_batch(load_audio(MODELS.parent / "examples" / "great_kiskadee_XC248744.ogg"))  # real spectrograms
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = SpeciesClassifier().session.run(None, {"spectrogram": x})[0]
    assert np.abs(ref - out).max() < 1e-3
