# CPU-only image for the species classifier API (also used as-is on Hugging Face Spaces)
FROM python:3.12-slim

# ffmpeg lets librosa decode formats libsndfile can't (e.g. m4a from phones)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (uid 1000, required by Hugging Face Spaces)
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app PATH=/home/app/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /home/app/service

COPY --chown=app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
COPY --chown=app models/efficientnet_b0.onnx models/labels.json ./models/
COPY --chown=app examples ./examples
RUN pip install --no-cache-dir --user --no-deps .

ENV MODEL_DIR=/home/app/service/models
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "birdclef_api.app:app", "--host", "0.0.0.0", "--port", "7860"]
