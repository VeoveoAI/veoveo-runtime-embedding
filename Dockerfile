# syntax=docker/dockerfile:1.27@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv
FROM python:3.14.6-slim-trixie@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144 AS base
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /opt/embedding
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY veoveo_runtime_embedding ./veoveo_runtime_embedding

FROM base AS check
COPY tests ./tests
# Base-only check is valid on Linux/arm64 and amd64 and never loads/downloads a model.
RUN uv run --frozen --no-sync python -m unittest discover -s tests -v

FROM base AS model
# Native upstream media decoding uses FFmpeg shared libraries on CPU; model inference stays CUDA.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg=7:7.1.5-0+deb13u1 \
    && rm -rf /var/lib/apt/lists/*
# Production Linux/amd64 resolves the official +cu130 wheels from the explicit index.
RUN uv sync --frozen --no-dev --no-install-project --extra model

FROM model AS runtime
ARG REVISION=uncommitted
ENV VEOVEO_EMBEDDING_DEVICE=cuda VEOVEO_EMBEDDING_BIND=0.0.0.0 VEOVEO_EMBEDDING_PORT=8804 VEOVEO_RUNTIME_REVISION=${REVISION} HF_HOME=/var/cache/huggingface
RUN useradd --system --uid 10001 --no-create-home embedding \
    && install -d -o embedding -g "$(id -gn embedding)" -m 0700 /sandbox
# Application code stays root-owned; the uninstalled project is imported explicitly.
ENV PYTHONPATH=/opt/embedding
WORKDIR /sandbox
USER embedding
EXPOSE 8804
ENTRYPOINT ["/opt/embedding/.venv/bin/python", "-m", "veoveo_runtime_embedding"]
LABEL org.opencontainers.image.title="veoveo-runtime-embedding" \
      org.opencontainers.image.source="https://github.com/VeoveoAI/veoveo-runtime-embedding" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0"
