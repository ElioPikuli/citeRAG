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

EXPOSE 7860
CMD ["python", "-m", "citerag.cli", "serve", "--host", "0.0.0.0", "--port", "7860"]
