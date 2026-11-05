# Hugging Face Space build: fetch CiteRAG from GitHub and build it with the repo's own Dockerfile steps.
# Rebuild the Space (Settings > Factory rebuild) to pick up new commits or project changes.
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

RUN git clone --depth 1 https://github.com/ElioPikuli/citeRAG.git . \
    && pip install --no-cache-dir --user . \
    && python scripts/fetch_projects.py \
    && python -m citerag.cli ingest

EXPOSE 7860
CMD ["python", "-m", "citerag.cli", "serve", "--host", "0.0.0.0", "--port", "7860"]
