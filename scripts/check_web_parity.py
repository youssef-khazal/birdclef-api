"""Check that the in-browser pipeline (web/classifier.js) matches the Python service.

    cd web && npm install && cd ..
    cp models/efficientnet_b0.onnx models/labels.json web/assets/ && mkdir -p web/assets/ort \
        && cp web/node_modules/onnxruntime-web/dist/{ort.wasm.min.js,ort-wasm-simd-threaded.mjs,ort-wasm-simd-threaded.wasm} web/assets/ort/
    python -m http.server 8765 &        # from the repository root
    python scripts/check_web_parity.py   # needs: pip install playwright && playwright install chromium

Compares, in headless Chromium: the bilinear resize against Pillow (random image), the first spectrogram
image of each example against Python, and the final predictions against the Python classifier.
"""

import asyncio
import sys

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

from birdclef_api.model import SpeciesClassifier
from birdclef_api.preprocessing import load_audio, spectrogram_image, split_into_segments

URL = "http://localhost:8765/web/tests/parity.html"
EXAMPLES = ["great_kiskadee_XC248744.ogg", "bananaquit_XC912389.ogg", "whistling_grass_frog_XC882794.ogg"]


async def main() -> None:
    clf = SpeciesClassifier()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(URL)
        await page.evaluate("window.ready")

        img = np.random.default_rng(0).integers(0, 256, (64, 626), dtype=np.uint8)
        js = np.array(await page.evaluate("([p, w, h]) => window.resize(p, w, h)", [img.ravel().tolist(), 626, 64]))
        pil = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
        resize_diff = int(np.abs(js.reshape(224, 224).astype(int) - pil).max())
        print(f"resize vs Pillow: max difference {resize_diff}")
        failures = ["resize differs from Pillow"] if resize_diff else []

        for name in EXAMPLES:
            r = await page.evaluate("(n) => window.runExample(n)", name)
            y = load_audio(f"examples/{name}")
            ref = np.asarray(spectrogram_image(split_into_segments(y)[0]), dtype=int)
            diff = np.abs(np.array(r["gray"]).reshape(ref.shape) - ref)
            py = clf.predict(y)["top_k"][0]
            js_top = r["result"]["top_k"][0]
            print(f"{name}: spectrogram pixels differing {np.mean(diff > 0):.2%} (max {diff.max()}); "
                  f"top-1 browser {js_top['common_name']} {js_top['probability']:.3f} vs Python {py['common_name']} {py['probability']:.3f}")
            if js_top["code"] != py["code"] or abs(js_top["probability"] - py["probability"]) > 0.01:
                failures.append(f"{name}: different prediction")
        await browser.close()
    if failures:
        sys.exit("FAILED: " + "; ".join(failures))
    print("browser and Python pipelines agree")


if __name__ == "__main__":
    asyncio.run(main())
