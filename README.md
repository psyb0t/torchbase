# Torchbase

[![CI](https://github.com/psyb0t/torchbase/actions/workflows/pipeline.yml/badge.svg?branch=main)](https://github.com/psyb0t/torchbase/actions/workflows/pipeline.yml)
[![version](https://raw.githubusercontent.com/psyb0t/torchbase/badges/version.svg)](https://github.com/psyb0t/torchbase/releases)
[![license](https://raw.githubusercontent.com/psyb0t/torchbase/badges/license.svg)](LICENSE)
[![Docker Pulls](https://img.shields.io/docker/pulls/psyb0t/torchbase?style=flat-square)](https://hub.docker.com/r/psyb0t/torchbase)

The heavy part of a PyTorch image, built once so every service does not waste hours rebuilding the same CUDA stack.

Torchbase publishes strict Python, Torch, and CUDA tuples. Pick the exact tuple your service needs. Do not point a production image at `latest` and hope the ABI works out.

## Use it

Published tags include the Torchbase release version. Replace `<release>` with a real version such as `v0.2.0`.

| Image | Python | PyTorch | CUDA |
| --- | --- | --- | --- |
| `psyb0t/torchbase:py3.12-torch2.14-v<release>-cpu` | 3.12 | `2.14.0+cpu` | None |
| `psyb0t/torchbase:py3.12-torch2.14-v<release>-cu126` | 3.12 | `2.14.0+cu126` | 12.6 |
| `psyb0t/torchbase:py3.12-torch2.14-v<release>-cu130` | 3.12 | `2.14.0+cu130` | 13.0 |
| `psyb0t/torchbase:py3.13-torch2.14-v<release>-cpu` | 3.13 | `2.14.0+cpu` | None |
| `psyb0t/torchbase:py3.13-torch2.14-v<release>-cu126` | 3.13 | `2.14.0+cu126` | 12.6 |
| `psyb0t/torchbase:py3.13-torch2.14-v<release>-cu130` | 3.13 | `2.14.0+cu130` | 13.0 |

All images put Torch in `/opt/torch-venv`, set `VIRTUAL_ENV` and `PATH`, and run as UID and GID `1000`.

```dockerfile
FROM psyb0t/torchbase:py3.12-torch2.14-v<release>-cu126

USER root
COPY --from=ghcr.io/astral-sh/uv:0.11.15@sha256:e590846f4776907b254ac0f44b5b380347af5d90d668138ca7938d1b0c2f98d3 /uv /usr/local/bin/uv
COPY requirements-provider.txt ./
RUN uv pip install --python /opt/torch-venv/bin/python --require-hashes -r requirements-provider.txt
USER 1000:1000
```

Torchbase owns Python, Torch, CUDA, and compiler bits. Your image owns its provider packages, model files, API, and tests.

CUDA 12.6 and CUDA 13.0 images need NVIDIA Container Toolkit at runtime. CUDA 12.x needs an NVIDIA driver from the 525 series or newer. CUDA 13.0 needs driver 580 or newer. Python 3.12 CUDA images use NVIDIA CUDA runtime bases. Python 3.13 CUDA images use the CUDA user-space libraries that ship with the hash-locked PyTorch wheel, then receive the host driver through NVIDIA Container Toolkit.

## Tuple matrix

`tuples.json` is the source of truth. Every entry names an exact builder image, runtime image, Python version, Torch backend, lock files, image tag shape, CPU or CUDA platform, and any runtime packages the CUDA base needs.

One generic Dockerfile and one GitHub Actions matrix build every entry. Adding Python, CUDA, or Torch releases means adding one tested tuple and its hash-locked requirements files. It does not mean copying a Dockerfile, a Make target, or a workflow job.

Each tuple's lock file is the dependency layer boundary. Docker and Buildx reuse that layer when the lock is unchanged, so a change outside the lock does not re-download PyTorch or CUDA packages.

## Build and verify

Docker and Make are the only host requirements.

```bash
make lint
make test
make test-coverage
make build-test
make build-test-cuda-gpu
```

`make build-test` builds and imports every registered tuple. `make build-test-cuda-gpu` builds and runs every CUDA tuple with `--gpus all`. It needs NVIDIA Container Toolkit and a usable NVIDIA GPU.

Use one tuple when you are working on a new one:

```bash
make model-lock TUPLE=py3.12-torch2.14-cpu
make build-test TUPLE=py3.12-torch2.14-cpu
```

`make ci-targets` prints the exact matrix consumed by the release workflow.

## Add a tuple

Add a hash-locked `.in` and `.txt` dependency pair, then add one object to `tuples.json`. Give it a new tag shape. Existing tuples never move in place.

Run the targeted lock and image tests before releasing it:

```bash
make model-lock TUPLE=your-tuple-id
make build-test TUPLE=your-tuple-id
```

CUDA entries also need `make build-test-cuda-gpu TUPLE=your-tuple-id` on a machine with a compatible driver. A derived service still needs its own image and GPU behavior tests.

## License

Torchbase code is WTFPL. PyTorch, CUDA, Python, and every dependency keep their own upstream licenses.
