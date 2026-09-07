# VoiceAgent container image — one image, two long-running services
# (scripts/chat_server.py on 8000, scripts/livekit_worker.py on 8080).
#
# Multi-stage: every pinned dependency ships a prebuilt cp312/aarch64 wheel
# EXCEPT llama-cpp-python (sdist -> cmake + C compiler). The builder stage
# carries build-essential/cmake and produces a wheelhouse; the final stage has
# NO compilers and installs from it with --no-index (smaller, no toolchain
# attack surface). requirements.txt is copied first so dependency resolution
# caches across source-only changes.
#
# The app runs FROM SOURCE (pyproject.toml declares no build backend): source
# is copied under /app with PYTHONPATH=/app/src — the package is never
# pip-installed. Secrets come from the environment at runtime; .env is
# gitignored and .dockerignore keeps it out of the build context entirely.

# --- stage 1: wheelhouse (the only stage that compiles anything) -----------
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# pip fetches wheels for everything that has one; llama-cpp-python builds
# from sdist here (needs cmake + g++, present above).
RUN pip wheel --no-cache-dir --wheel-dir /wheelhouse -r requirements.txt

# --- stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install from the prebuilt wheelhouse only — no compilation in this stage.
COPY --from=builder /wheelhouse /wheelhouse
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheelhouse \
        -r requirements.txt \
    && rm -rf /wheelhouse

# Non-root runtime user; source and data dirs are owned by it.
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/
# Runtime data: tenant bundles, platform policy fallback, knowledge docs.
# data/models (GB-scale local model cache), data/index and data/out are
# deliberately NOT baked in — see .dockerignore and DEPLOY.md.
COPY --chown=appuser:appuser data/tenants/ data/tenants/
COPY --chown=appuser:appuser data/policies/ data/policies/
COPY --chown=appuser:appuser data/knowledge/ data/knowledge/
COPY --chown=appuser:appuser pyproject.toml ./

# Mutable state lives on volumes (docker-compose.yml): chat memory DB
# (data/out/memory.db), RAG chunk cache (data/index/chunks.pkl), and any
# VOICEAGENT_AUDIT_DB / VOICEAGENT_MEMORY_DB paths pointed in here.
RUN mkdir -p data/out data/index && chown -R appuser:appuser data

USER appuser

EXPOSE 8000 8080

# Default: the governed demo HTTP server. The LiveKit worker overrides the
# command (docker-compose.yml: `worker` service).
CMD ["python", "scripts/chat_server.py", "8000", "0.0.0.0"]
