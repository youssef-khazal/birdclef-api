"""Export the trained EfficientNet-B0 (PyTorch) to ONNX, then quantize it to INT8.

    python scripts/export_onnx.py \
        --run-dir ../birdclef-2025/runs/efficientnet_b0_recording_seed42 \
        --taxonomy data/taxonomy.csv \
        --calibration-dir data/calibration_pngs \
        --out-dir models

Writes:
  models/efficientnet_b0.onnx        FP32, same weights as the PyTorch checkpoint
  models/efficientnet_b0_int8.onnx   INT8 (static quantization, per-channel weights)
  models/labels.json                 class index -> species code, common and scientific name

The calibration images must come from the *training* split (never the validation split
used for the benchmark). Requires the export extras: pip install -e ".[export]"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from torchvision import models

from birdclef_api.preprocessing import IMG_SIZE, image_to_input


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def export_fp32(run_dir: Path, out_path: Path) -> list[str]:
    classes = json.loads((run_dir / "classes.json").read_text())
    model = build_efficientnet_b0(len(classes))
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(out_path), input_names=["spectrogram"], output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    return classes


class PngCalibrationReader(CalibrationDataReader):
    """Feeds training-set spectrogram PNGs to the quantizer, a few at a time."""

    def __init__(self, png_dir: Path, limit: int = 300, batch: int = 8, seed: int = 0):
        files = sorted(png_dir.rglob("*.png"))
        random.Random(seed).shuffle(files)
        self.batches = [files[i : i + batch] for i in range(0, min(limit, len(files)), batch)]
        self._it = iter(self.batches)

    def get_next(self):
        chunk = next(self._it, None)
        if chunk is None:
            return None
        return {"spectrogram": np.stack([image_to_input(Image.open(f)) for f in chunk])}


def quantize_int8(fp32_path: Path, int8_path: Path, calibration_dir: Path) -> None:
    prep = fp32_path.with_suffix(".prep.onnx")
    quant_pre_process(str(fp32_path), str(prep))
    quantize_static(
        str(prep), str(int8_path), PngCalibrationReader(calibration_dir),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    )
    prep.unlink()


def write_labels(classes: list[str], taxonomy_csv: Path, out_path: Path) -> None:
    with open(taxonomy_csv, newline="", encoding="utf-8") as f:
        tax = {row["primary_label"]: row for row in csv.DictReader(f)}
    labels = [
        {"code": c, "common_name": tax.get(c, {}).get("common_name", c),
         "scientific_name": tax.get(c, {}).get("scientific_name", ""), "group": tax.get(c, {}).get("class_name", "")}
        for c in classes
    ]
    out_path.write_text(json.dumps(labels, indent=1, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--taxonomy", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = a.out_dir / "efficientnet_b0.onnx"
    classes = export_fp32(a.run_dir, fp32)
    quantize_int8(fp32, a.out_dir / "efficientnet_b0_int8.onnx", a.calibration_dir)
    write_labels(classes, a.taxonomy, a.out_dir / "labels.json")
    print(f"Exported {len(classes)} classes to {a.out_dir}")


if __name__ == "__main__":
    main()
