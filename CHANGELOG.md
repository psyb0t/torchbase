# Changelog

All notable changes per release. Versions follow [SemVer](https://semver.org/).

## v0.1.0

Initial release.

- Added a digest-pinned PyTorch base-image matrix with Python 3.12 and 3.13 CPU, CUDA 12.6, and CUDA 13.0 tuples.
- Added one generic Dockerfile, one validated tuple registry, and generated Docker workflow targets for future Python, CUDA, and Torch combinations.
- Added locked CPU and CUDA dependencies, non-root runtime images, local CPU and GPU image verification, and Docker Buildx cache export.
