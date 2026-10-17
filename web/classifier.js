// In-browser version of the species classifier: audio -> spectrogram -> ONNX model, all client-side.
// The preprocessing reproduces the Python pipeline (src/birdclef_api/preprocessing.py) step by step:
// librosa log-Mel, matplotlib's 8-bit grayscale mapping, Pillow's bilinear resize, [-1, 1] scaling.

const SR = 32000;
const SEG = 5 * SR;
const IMG = 224;

let params = null;   // mel filterbank + matplotlib gray LUT (exported from Python)
let labels = null;
let session = null;

export async function load(base = "assets/") {
  ort.env.wasm.numThreads = 1;                // no cross-origin isolation on static hosting
  ort.env.wasm.wasmPaths = new URL(base + "ort/", location.href).href;
  [params, labels] = await Promise.all([
    fetch(base + "audio_params.json").then((r) => r.json()),
    fetch(base + "labels.json").then((r) => r.json()),
  ]);
  session = await ort.InferenceSession.create(base + "efficientnet_b0.onnx", { executionProviders: ["wasm"] });
}

// ---------------------------------------------------------------- audio
export async function decode(arrayBuffer) {
  // Decoding with a 32 kHz context makes the browser resample to 32 kHz.
  const ctx = new OfflineAudioContext(1, 1, SR);
  const buf = await ctx.decodeAudioData(arrayBuffer);
  const n = buf.length, out = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return out;
}

export function segments(y) {
  const segs = [];
  for (let i = 0; i + SEG <= y.length; i += SEG) segs.push(y.subarray(i, i + SEG));
  const done = segs.length * SEG;
  if (y.length > done || segs.length === 0) {
    const last = new Float32Array(SEG);
    last.set(y.subarray(done));
    segs.push(last);
  }
  return segs;
}

// ---------------------------------------------------------------- spectrogram (librosa)
function fftReal(re, im) {               // in-place radix-2 FFT, length power of 2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k + len / 2] = re[i + k] - ar; im[i + k + len / 2] = im[i + k] - ai;
        re[i + k] += ar; im[i + k] += ai;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

export function logMel(seg) {
  const { n_fft: N, hop, n_mels: M, mel } = params;
  const pad = N / 2, frames = 1 + Math.floor(seg.length / hop);
  const padded = new Float32Array(seg.length + N);          // center=True, pad_mode="constant"
  padded.set(seg, pad);
  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);  // periodic Hann
  const S = new Float64Array(M * frames);                    // [mel, frame]
  const re = new Float64Array(N), im = new Float64Array(N), pow = new Float64Array(N / 2 + 1);
  for (let t = 0; t < frames; t++) {
    for (let i = 0; i < N; i++) { re[i] = padded[t * hop + i] * win[i]; im[i] = 0; }
    fftReal(re, im);
    for (let k = 0; k <= N / 2; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < M; m++) {
      const { start, weights } = mel[m];
      let s = 0;
      for (let j = 0; j < weights.length; j++) s += weights[j] * pow[start + j];
      S[m * frames + t] = s;
    }
  }
  let ref = 0;
  for (const v of S) ref = Math.max(ref, v);
  const refDb = 10 * Math.log10(Math.max(1e-10, ref));
  const db = new Float32Array(S.length);
  let top = -Infinity;
  for (let i = 0; i < S.length; i++) { db[i] = 10 * Math.log10(Math.max(1e-10, S[i])) - refDb; top = Math.max(top, db[i]); }
  for (let i = 0; i < db.length; i++) db[i] = Math.max(db[i], top - 80);   // top_db = 80
  return { db, width: frames, height: M };
}

// ---------------------------------------------------------------- image (matplotlib imsave, cmap="gray")
export function toGray8(spec) {
  const { db } = spec;
  let lo = Infinity, hi = -Infinity;
  for (const v of db) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const f = Math.fround, range = f(hi - lo), lut = params.gray_lut;
  const out = new Uint8Array(db.length);
  for (let i = 0; i < db.length; i++) {
    let x = f(f(f(db[i] - lo) / range) * 256);              // float32 arithmetic, as in matplotlib
    let idx = x >= 256 ? 255 : Math.floor(x);
    out[i] = lut[Math.max(0, Math.min(255, idx))];
  }
  return { pixels: out, width: spec.width, height: spec.height };   // row 0 = lowest mel band
}

// ---------------------------------------------------------------- Pillow bilinear resize (Resample.c, 8-bit)
function coeffs(inSize, outSize) {
  const scale = inSize / outSize, filterscale = Math.max(scale, 1), support = filterscale;
  const out = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    const k = [];
    let ww = 0;
    for (let x = 0; x < xmax - xmin; x++) {
      const t = Math.abs((x + xmin - center + 0.5) / filterscale);
      const w = t < 1 ? 1 - t : 0;
      k.push(w); ww += w;
    }
    out.push({ xmin, k: k.map((w) => { const v = ww ? w / ww : w; return Math.trunc((v < 0 ? -0.5 : 0.5) + v * 4194304); }) });
  }
  return out;
}

const clip8 = (v) => Math.min(255, Math.max(0, Math.floor(v / 4194304)));

export function resizeBilinear(img, W = IMG, H = IMG) {
  const { pixels, width, height } = img;
  const ch = coeffs(width, W), cv = coeffs(height, H);
  const tmp = new Uint8Array(W * height);                  // horizontal pass first, as Pillow
  for (let y = 0; y < height; y++)
    for (let x = 0; x < W; x++) {
      const { xmin, k } = ch[x];
      let ss = 2097152;
      for (let i = 0; i < k.length; i++) ss += pixels[y * width + xmin + i] * k[i];
      tmp[y * W + x] = clip8(ss);
    }
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const { xmin, k } = cv[y];
    for (let x = 0; x < W; x++) {
      let ss = 2097152;
      for (let i = 0; i < k.length; i++) ss += tmp[(xmin + i) * W + x] * k[i];
      out[y * W + x] = clip8(ss);
    }
  }
  return out;
}

// ---------------------------------------------------------------- model

export function toInput(resized) {
  const plane = IMG * IMG, x = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const v = (resized[i] / 255 - 0.5) / 0.5;
    x[i] = v; x[plane + i] = v; x[2 * plane + i] = v;
  }
  return x;
}

function softmax(row) {
  let m = -Infinity; for (const v of row) m = Math.max(m, v);
  const e = row.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export async function classify(y, topK = 5) {
  const t0 = performance.now();
  const segs = segments(y);
  const inputs = segs.map((s) => toInput(resizeBilinear(toGray8(logMel(s)))));
  const t1 = performance.now();
  const probs = [];
  for (const x of inputs) {
    const out = await session.run({ spectrogram: new ort.Tensor("float32", x, [1, 3, IMG, IMG]) });
    probs.push(softmax(Array.from(out.logits.data)));
  }
  const t2 = performance.now();
  const mean = probs[0].map((_, j) => probs.reduce((a, p) => a + p[j], 0) / probs.length);
  const rank = (p, k) => p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k)
    .map(([v, i]) => ({ ...labels[i], probability: v }));
  return {
    duration_seconds: +(y.length / SR).toFixed(2),
    segments: segs.length,
    top_k: rank(mean, topK),
    per_segment: probs.map((p, k) => ({ start_seconds: k * 5, ...rank(p, 1)[0] })),
    timing_ms: { spectrograms: Math.round(t1 - t0), model: Math.round(t2 - t1) },
  };
}
