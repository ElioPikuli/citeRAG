# CiteRAG, self-contained: the corpus, the embedding model and the index are all built into the image,
# so the container starts without downloads and needs no API key (it answers in offline mode unless
# ANTHROPIC_API_KEY is set). Runs as uid 1000, as Hugging Face Spaces requires.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/models
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user citerag ./citerag
RUN pip install --no-cache-dir --user .

COPY --chown=user . .
RUN python scripts/fetch_projects.py && python -m citerag.cli ingest

# Runtime only (the index above was built with all cores): one ONNX thread is ~10x faster on a CPU-quota host
# such as Render's free 0.1 CPU (measured: 5-10 s -> 0.6 s per question).
ENV CITERAG_THREADS=1 OMP_NUM_THREADS=1

# Hosts like Render set $PORT; Hugging Face Spaces and local runs use 7860.
EXPOSE 7860
CMD ["sh", "-c", "exec python -m citerag.cli serve --host 0.0.0.0 --port ${PORT:-7860}"]
