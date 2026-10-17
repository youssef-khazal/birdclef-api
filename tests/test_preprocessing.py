from pathlib import Path

import numpy as np
from PIL import Image

from birdclef_api.preprocessing import (
    IMG_SIZE, SAMPLE_RATE, audio_to_batch, image_to_input, load_audio, spectrogram_image, split_into_segments,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "great_kiskadee_XC248744.ogg"
TRAINING_PNG = ROOT / "tests" / "data" / "great_kiskadee_XC248744_seg0_training.png"


def test_spectrogram_is_pixel_identical_to_training_data():
    """The first segment of this recording must give exactly the PNG the model was trained on."""
    seg0 = split_into_segments(load_audio(EXAMPLE))[0]
    ours = np.asarray(spectrogram_image(seg0), dtype=int)
    stored = np.asarray(Image.open(TRAINING_PNG).convert("L"), dtype=int)
    assert ours.shape == stored.shape == (64, 626)
    assert np.array_equal(ours, stored)


def test_segments_are_five_seconds_and_last_is_padded():
    y = np.ones(int(SAMPLE_RATE * 12.5), dtype=np.float32)
    segs = split_into_segments(y)
    assert len(segs) == 3 and all(len(s) == 5 * SAMPLE_RATE for s in segs)
    assert segs[-1][: int(2.5 * SAMPLE_RATE)].all() and not segs[-1][int(2.5 * SAMPLE_RATE) :].any()


def test_short_audio_gives_one_segment():
    assert len(split_into_segments(np.ones(SAMPLE_RATE, dtype=np.float32))) == 1


def test_model_input_shape_and_range():
    x = image_to_input(Image.fromarray(np.arange(64 * 626, dtype=np.uint32).reshape(64, 626).astype(np.uint8)))
    assert x.shape == (3, IMG_SIZE, IMG_SIZE) and x.dtype == np.float32
    assert -1.0 <= x.min() and x.max() <= 1.0
    assert np.array_equal(x[0], x[2])


def test_audio_from_bytes_matches_audio_from_path():
    assert np.array_equal(load_audio(EXAMPLE.read_bytes()), load_audio(EXAMPLE))


def test_batch_has_one_item_per_segment():
    y = load_audio(EXAMPLE)
    assert audio_to_batch(y).shape == (len(split_into_segments(y)), 3, IMG_SIZE, IMG_SIZE)
