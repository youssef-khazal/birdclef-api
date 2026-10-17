"""Export the constants the in-browser preprocessing needs, straight from librosa and matplotlib.

    python scripts/export_web_params.py   # writes web/assets/audio_params.json

- the 64 x 513 Mel filterbank of librosa.feature.melspectrogram (32 kHz, n_fft 1024), stored sparsely
- the exact 8-bit lookup table of matplotlib's "gray" colormap (it is not the identity: 24 of 256 levels differ)
"""

import json
from pathlib import Path

import librosa
import matplotlib
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "web" / "assets" / "audio_params.json"

fb = librosa.filters.mel(sr=32000, n_fft=1024, n_mels=64).astype(np.float32)
mel = []
for row in fb:
    nz = np.nonzero(row)[0]
    mel.append({"start": int(nz[0]), "weights": [float(w) for w in row[nz[0] : nz[-1] + 1]]})
lut = matplotlib.colormaps["gray"](np.arange(256), bytes=True)[:, 0].astype(int).tolist()
OUT.write_text(json.dumps({"sr": 32000, "n_fft": 1024, "hop": 256, "n_mels": 64, "mel": mel, "gray_lut": lut}))
print(f"wrote {OUT}")
