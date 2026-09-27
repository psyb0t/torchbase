# syntax=docker/dockerfile:1.7

ARG PYTHON_BUILDER_IMAGE
ARG RUNTIME_IMAGE

FROM ${PYTHON_BUILDER_IMAGE} AS torch-builder

ARG EXPECTED_CUDA_VERSION
ARG EXPECTED_PYTHON_VERSION
ARG EXPECTED_TORCH_VERSION
ARG REQUIREMENTS_FILE
ARG TORCH_BACKEND

ENV UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.11.15@sha256:e590846f4776907b254ac0f44b5b380347af5d90d668138ca7938d1b0c2f98d3 /uv /usr/local/bin/uv

WORKDIR /build

COPY ${REQUIREMENTS_FILE} ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/torch-venv \
    && uv pip install --python /opt/torch-venv/bin/python --require-hashes --torch-backend ${TORCH_BACKEND} -r requirements.txt \
    && /opt/torch-venv/bin/python -c "import sys, torch; assert f'{sys.version_info.major}.{sys.version_info.minor}' == '${EXPECTED_PYTHON_VERSION}'; assert torch.__version__ == '${EXPECTED_TORCH_VERSION}'; assert (torch.version.cuda or 'none') == '${EXPECTED_CUDA_VERSION}'"

FROM ${RUNTIME_IMAGE}

ARG RUNTIME_APT_PACKAGES
ARG RUNTIME_PYTHON

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/torch-venv \
    PATH=/opt/torch-venv/bin:$PATH

LABEL org.opencontainers.image.description="Pinned Python, PyTorch, and CUDA base for psyb0t containers" \
    org.opencontainers.image.source="https://github.com/psyb0t/torchbase"

COPY --from=torch-builder /opt/torch-venv /opt/torch-venv

RUN if [ -n "${RUNTIME_APT_PACKAGES}" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends $(printf '%s' "${RUNTIME_APT_PACKAGES}" | tr ',' ' ') \
        && rm --recursive --force /var/lib/apt/lists/*; \
    fi \
    && if [ -n "${RUNTIME_PYTHON}" ]; then \
        ln --symbolic --force "${RUNTIME_PYTHON}" /usr/local/bin/python3 \
        && ln --symbolic --force /usr/local/bin/python3 /usr/local/bin/python; \
    fi \
    && (getent group 1000 >/dev/null || groupadd --gid 1000 torchbase) \
    && (getent passwd 1000 >/dev/null || useradd --create-home --uid 1000 --gid 1000 --shell /usr/sbin/nologin torchbase)

USER 1000:1000
WORKDIR /work
