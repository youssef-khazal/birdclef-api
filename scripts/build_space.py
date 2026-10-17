"""Assemble the static Hugging Face Space (in-browser demo) into one folder.

    cd web && npm install && cd ..
    python scripts/build_space.py --out dist/space

Copies web/ (page + JavaScript preprocessing), the ONNX model, the labels, the example
recordings and the ONNX Runtime Web files, and writes the Space README (sdk: static).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORT_FILES = ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]

SPACE_README = """---
title: Species Sound Classifier
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Identify 206 bird, frog, insect and mammal species by sound
---

# Species Sound Classifier

Upload a field recording and an EfficientNet-B0 model identifies which of 206 species from the BirdCLEF+ 2025
dataset it hears. The model runs entirely in your browser (ONNX Runtime Web); audio is never uploaded.

Source code, Docker API, benchmark and tests: [github.com/youssef-khazal/birdclef-api](https://github.com/youssef-khazal/birdclef-api)
· Author: Youssef Khazal
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "space")
    out = p.parse_args().out
    if out.exists():
        shutil.rmtree(out)
    (out / "assets" / "ort").mkdir(parents=True)
    for f in ["index.html", "classifier.js"]:
        shutil.copy(ROOT / "web" / f, out / f)
    shutil.copy(ROOT / "web" / "assets" / "audio_params.json", out / "assets")
    shutil.copy(ROOT / "models" / "efficientnet_b0.onnx", out / "assets")
    shutil.copy(ROOT / "models" / "labels.json", out / "assets")
    ort_dist = ROOT / "web" / "node_modules" / "onnxruntime-web" / "dist"
    for f in ORT_FILES:
        shutil.copy(ort_dist / f, out / "assets" / "ort" / f)
    shutil.copytree(ROOT / "examples", out / "examples")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    print(f"Space ready in {out}")


if __name__ == "__main__":
    main()
