import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from birdclef_api.app import app

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
client = TestClient(app)


def post(name: str, data: bytes, **params):
    return client.post("/predict", files={"file": (name, data, "application/octet-stream")}, params=params)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["classes"] == 206


def test_species_list():
    species = client.get("/species").json()
    assert len(species) == 206 and {"code", "common_name", "scientific_name", "group"} <= set(species[0])


def test_demo_page_is_served():
    assert "Species Sound Classifier" in client.get("/").text


@pytest.mark.parametrize("file, expected", [
    ("great_kiskadee_XC248744.ogg", "Great Kiskadee"),
    ("bananaquit_XC912389.ogg", "Bananaquit"),
    ("whistling_grass_frog_XC882794.ogg", "Whistling Grass Frog"),
])
def test_examples_are_recognised(file, expected):
    r = post(file, (EXAMPLES / file).read_bytes())
    assert r.status_code == 200
    body = r.json()
    assert body["top_k"][0]["common_name"] == expected
    assert len(body["top_k"]) == 5
    assert abs(sum(p["probability"] for p in body["top_k"]) - 1) < 1.0  # top-5 of a softmax
    assert len(body["per_segment"]) == body["segments"]


def test_wav_upload_and_top_k_parameter():
    buf = io.BytesIO()
    sf.write(buf, np.random.default_rng(0).normal(0, 0.1, 44_100 * 3).astype(np.float32), 44_100, format="WAV")
    r = post("noise.wav", buf.getvalue(), top_k=3)
    assert r.status_code == 200 and len(r.json()["top_k"]) == 3 and r.json()["segments"] == 1


@pytest.mark.parametrize("data, status", [(b"", 400), (b"not audio at all", 415)])
def test_bad_uploads_are_rejected(data, status):
    assert post("bad.ogg", data).status_code == status


def test_example_route_does_not_serve_other_files():
    assert client.get("/examples/..%2F..%2Fpyproject.toml").status_code == 404
