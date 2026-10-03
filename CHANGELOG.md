# Changelog

All notable changes per release. Versions follow [SemVer](https://semver.org/).

## v0.1.1

- Add Python 3.12, Torch 2.5.1 CPU and CUDA 12.4 base tuples for Audiolla, with separate hash-locked dependencies and image tag shapes.
- Honor tuple selection in GPU build tests and verify real tensor operations on CPU and CUDA.
- Document matching downstream Torchaudio installation and the retained Torch checkpoint-loading vulnerability.
- Grant the reusable code workflow permission to upload security results.

## v0.1.0

Initial release.

- Added a digest-pinned PyTorch base-image matrix with Python 3.12 and 3.13 CPU, CUDA 12.6, and CUDA 13.0 tuples.
- Added one generic Dockerfile, one validated tuple registry, and generated Docker workflow targets for future Python, CUDA, and Torch combinations.
- Added locked CPU and CUDA dependencies, non-root runtime images, local CPU and GPU image verification, and Docker Buildx cache export.
