# syntax=docker/dockerfile:1

FROM python:3.11-slim AS cpu
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip==24.3.1 \
    && python -m pip install -e ".[dev]"
COPY . .
CMD ["python", "-c", "import medguard; print(medguard.__version__)"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# GPU runtime dependency installation is intentionally deferred to later phases.
# See DECISIONS.md for the Phase 0A review-fix deferral record.
COPY . .
CMD ["bash", "-lc", "echo 'MEDGUARD-CXR GPU target placeholder; deferred until Phase 1'"]
