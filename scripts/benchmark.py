"""Compare PyTorch, ONNX FP32 and ONNX INT8 on accuracy, speed and size (CPU).

    python scripts/benchmark.py --val-dir data/val_sample --run-dir <birdclef-2025 run> \
        --audio examples/great_kiskadee_XC248744.ogg --out results/benchmark.json

Accuracy is measured on spectrograms from the held-out validation split of birdclef-2025
(recording-level split, seed 42); the PyTorch reference uses the exact training transform.
Latency is measured single-threaded and with all cores, batch of 1 segment, after warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from birdclef_api.preprocessing import audio_to_batch, image_to_input, load_audio


def onnx_session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def timed(fn, warmup: int = 5, runs: int = 50) -> float:
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t) / runs * 1000  # ms


def topk_acc(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([yy in t for yy, t in zip(y, top)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-dir", type=Path, required=True, help="<species>/<file>.png from the validation split")
    p.add_argument("--run-dir", type=Path, required=True, help="birdclef-2025 run folder (best.pth, classes.json)")
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--audio", type=Path, required=True, help="a recording for the end-to-end timing")
    p.add_argument("--out", type=Path, default=Path("results/benchmark.json"))
    a = p.parse_args()

    import torch
    from torchvision import transforms as T

    try:
        from export_onnx import build_efficientnet_b0  # python scripts/benchmark.py
    except ImportError:
        from scripts.export_onnx import build_efficientnet_b0

    classes = json.loads((a.run_dir / "classes.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    files = sorted(a.val_dir.rglob("*.png"))
    y = np.array([idx[f.parent.name] for f in files])

    # PyTorch reference: exact training transform on the stored PNGs
    model = build_efficientnet_b0(len(classes))
    model.load_state_dict(torch.load(a.run_dir / "best.pth", map_location="cpu", weights_only=True))
    model.eval()
    tf = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    with torch.no_grad():
        ref = np.concatenate([
            model(torch.stack([tf(Image.open(f)) for f in files[i : i + 64]])).numpy() for i in range(0, len(files), 64)
        ])
    x = np.stack([image_to_input(Image.open(f)) for f in files])
    # The NumPy/PIL preprocessing used by the API must match the torchvision training transform
    max_input_diff = float(np.abs(x[:50] - np.stack([tf(Image.open(f)).numpy() for f in files[:50]])).max())

    cores = os.cpu_count() or 1
    results = {"n_segments": len(files), "n_species": int(len(set(y))), "cpu": platform.processor() or platform.machine(),
               "cores": cores, "max_input_difference_vs_torchvision": max_input_diff, "models": {}}

    def add(name, logits, size_mb, latency_1, latency_all, e2e_ms):
        results["models"][name] = {
            "top1": topk_acc(logits, y, 1), "top5": topk_acc(logits, y, 5),
            "agreement_with_pytorch_top1": float(np.mean(logits.argmax(1) == ref.argmax(1))),
            "size_mb": size_mb, "latency_ms_1_thread": latency_1, "latency_ms_all_cores": latency_all,
            "end_to_end_ms_30s_recording": e2e_ms,
        }

    one = torch.from_numpy(x[:1])
    torch.set_num_threads(1)
    with torch.no_grad():
        l1 = timed(lambda: model(one))
    torch.set_num_threads(cores)
    with torch.no_grad():
        la = timed(lambda: model(one))
    add("pytorch_fp32", ref, (a.run_dir / "best.pth").stat().st_size / 1e6, l1, la, None)

    audio = load_audio(a.audio)[: 30 * 32_000]
    for name, fname in [("onnx_fp32", "efficientnet_b0.onnx"), ("onnx_int8", "efficientnet_b0_int8.onnx")]:
        path = a.models_dir / fname
        s_all = onnx_session(path, cores)
        logits = np.concatenate([s_all.run(None, {"spectrogram": x[i : i + 64]})[0] for i in range(0, len(x), 64)])
        s1 = onnx_session(path, 1)
        l1 = timed(lambda: s1.run(None, {"spectrogram": x[:1]}))
        la = timed(lambda: s_all.run(None, {"spectrogram": x[:1]}))
        e2e = timed(lambda: s_all.run(None, {"spectrogram": audio_to_batch(audio)}), warmup=2, runs=10)
        add(name, logits, path.stat().st_size / 1e6, l1, la, e2e)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
