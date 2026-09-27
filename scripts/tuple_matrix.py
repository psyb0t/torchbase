"""Validate and render the immutable Torchbase image tuple registry."""

from __future__ import annotations

import argparse
import json
import re

# Subprocess calls use argument vectors assembled from validated tuple data.
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "tuples.json"
DEFAULT_IMAGE_NAME = "psyb0t/torchbase"
DEFAULT_DOCKERFILE = "Dockerfile"
IMAGE_PATTERN = re.compile(r"^[^\s@]+@sha256:[a-f0-9]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
TAG_PART_PATTERN = re.compile(r"^[a-z0-9.-]+$")
PACKAGE_PATTERN = re.compile(r"^[A-Za-z0-9.+:=-]+$")
PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
PLATFORM_PATTERN = re.compile(r"^linux/[a-z0-9]+(?:,linux/[a-z0-9]+)*$")
CUDA_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")
PYTHON_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")
SELECTION_ALL = "all"
SELECTION_CPU = "cpu"
SELECTION_CUDA = "cuda"
# The path is a Docker container-private tmpfs mount, not a host path.
CONTAINER_TMPFS = "/tmp:rw,noexec,nosuid,size=8m"  # nosec B108


class RegistryError(ValueError):
    """Raised when a tuple registry cannot safely drive a build."""


@dataclass(frozen=True)
class TupleSpec:
    identifier: str
    tag_prefix: str
    tag_suffix: str
    local_tag: str
    platforms: str
    python_version: str
    python_builder_image: str
    runtime_image: str
    runtime_apt_packages: tuple[str, ...]
    runtime_python: str
    requirements_input: str
    requirements_lock: str
    torch_backend: str
    torch_version: str
    cuda_version: str

    @property
    def is_cuda(self) -> bool:
        return self.torch_backend != SELECTION_CPU

    @property
    def image_tag(self) -> str:
        return f"{DEFAULT_IMAGE_NAME}:{self.local_tag}"

    def build_args(self) -> dict[str, str]:
        return {
            "PYTHON_BUILDER_IMAGE": self.python_builder_image,
            "EXPECTED_PYTHON_VERSION": self.python_version,
            "RUNTIME_IMAGE": self.runtime_image,
            "RUNTIME_APT_PACKAGES": ",".join(self.runtime_apt_packages),
            "RUNTIME_PYTHON": self.runtime_python,
            "TORCH_BACKEND": self.torch_backend,
            "REQUIREMENTS_FILE": self.requirements_lock,
            "EXPECTED_TORCH_VERSION": self.torch_version,
            "EXPECTED_CUDA_VERSION": self.cuda_version,
        }


def fail(message: str) -> RegistryError:
    return RegistryError(message)


def require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise fail(f"{field} must be a non-empty string")
    return value


def require_optional_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise fail(f"{field} must be a string")
    return value


def require_safe_identifier(value: Any, field: str) -> str:
    validated_value = require_string(value, field)
    if not IDENTIFIER_PATTERN.fullmatch(validated_value):
        raise fail(f"{field} is not a safe image identifier: {validated_value!r}")
    return validated_value


def require_tag_part(value: Any, field: str) -> str:
    validated_value = require_string(value, field)
    if not TAG_PART_PATTERN.fullmatch(validated_value):
        raise fail(f"{field} is not a safe image tag part: {validated_value!r}")
    return validated_value


def require_image(value: Any, field: str) -> str:
    validated_value = require_string(value, field)
    if not IMAGE_PATTERN.fullmatch(validated_value):
        raise fail(f"{field} must be digest-pinned: {validated_value!r}")
    return validated_value


def require_relative_path(value: Any, field: str, suffix: str) -> str:
    validated_value = require_string(value, field)
    candidate = Path(validated_value)
    if (
        not PATH_PATTERN.fullmatch(validated_value)
        or candidate.is_absolute()
        or ".." in candidate.parts
        or not validated_value.endswith(suffix)
    ):
        raise fail(
            f"{field} must be a safe relative {suffix} path: {validated_value!r}"
        )
    return validated_value


def require_packages(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and PACKAGE_PATTERN.fullmatch(item) for item in value
    ):
        raise fail("runtime_apt_packages must be an array of safe package names")
    return tuple(value)


def parse_spec(value: Any) -> TupleSpec:
    if not isinstance(value, dict):
        raise fail("each tuple must be an object")
    backend = require_string(value.get("torch_backend"), "torch_backend")
    cuda_version = require_string(value.get("cuda_version"), "cuda_version")
    if backend == SELECTION_CPU:
        if cuda_version != "none":
            raise fail("CPU tuples must set cuda_version to 'none'")
    elif not re.fullmatch(r"cu\d+", backend) or not CUDA_VERSION_PATTERN.fullmatch(
        cuda_version
    ):
        raise fail("CUDA tuples need a cuNNN backend and a MAJOR.MINOR cuda_version")
    platforms = require_string(value.get("platforms"), "platforms")
    if not PLATFORM_PATTERN.fullmatch(platforms):
        raise fail(f"platforms is invalid: {platforms!r}")
    python_version = require_string(value.get("python_version"), "python_version")
    if not PYTHON_VERSION_PATTERN.fullmatch(python_version):
        raise fail(f"python_version must be MAJOR.MINOR: {python_version!r}")
    runtime_python = require_optional_string(
        value.get("runtime_python"), "runtime_python"
    )
    if runtime_python and (
        not runtime_python.startswith("/") or not PATH_PATTERN.fullmatch(runtime_python)
    ):
        raise fail(f"runtime_python must be an absolute safe path: {runtime_python!r}")
    return TupleSpec(
        identifier=require_safe_identifier(value.get("id"), "id"),
        tag_prefix=require_tag_part(value.get("tag_prefix"), "tag_prefix"),
        tag_suffix=require_tag_part(value.get("tag_suffix"), "tag_suffix"),
        local_tag=require_safe_identifier(value.get("local_tag"), "local_tag"),
        platforms=platforms,
        python_version=python_version,
        python_builder_image=require_image(
            value.get("python_builder_image"), "python_builder_image"
        ),
        runtime_image=require_image(value.get("runtime_image"), "runtime_image"),
        runtime_apt_packages=require_packages(value.get("runtime_apt_packages")),
        runtime_python=runtime_python,
        requirements_input=require_relative_path(
            value.get("requirements_input"), "requirements_input", ".in"
        ),
        requirements_lock=require_relative_path(
            value.get("requirements_lock"), "requirements_lock", ".txt"
        ),
        torch_backend=backend,
        torch_version=require_string(value.get("torch_version"), "torch_version"),
        cuda_version=cuda_version,
    )


def load_specs(registry_path: Path) -> tuple[Path, tuple[TupleSpec, ...]]:
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise fail(f"tuple registry not found: {registry_path}") from error
    except json.JSONDecodeError as error:
        raise fail(f"invalid tuple registry JSON: {error}") from error
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("tuples"), list)
        or not document["tuples"]
    ):
        raise fail("tuple registry must contain a non-empty tuples array")
    specs = tuple(parse_spec(value) for value in document["tuples"])
    identifiers = [spec.identifier for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise fail("tuple registry contains duplicate ids")
    local_tags = [spec.local_tag for spec in specs]
    if len(local_tags) != len(set(local_tags)):
        raise fail("tuple registry contains duplicate local tags")
    release_tags = [f"{spec.tag_prefix}latest{spec.tag_suffix}" for spec in specs]
    if len(release_tags) != len(set(release_tags)):
        raise fail("tuple registry contains duplicate release tag shapes")
    root = registry_path.parent.resolve()
    for spec in specs:
        if not (root / spec.requirements_input).is_file():
            raise fail(
                f"requirements input missing for {spec.identifier}: {spec.requirements_input}"
            )
        if not (root / spec.requirements_lock).is_file():
            raise fail(
                f"requirements lock missing for {spec.identifier}: {spec.requirements_lock}"
            )
    return root, specs


def select_specs(specs: tuple[TupleSpec, ...], selection: str) -> tuple[TupleSpec, ...]:
    if selection == SELECTION_ALL:
        return specs
    if selection == SELECTION_CPU:
        return tuple(spec for spec in specs if not spec.is_cuda)
    if selection == SELECTION_CUDA:
        return tuple(spec for spec in specs if spec.is_cuda)
    for spec in specs:
        if spec.identifier == selection:
            return (spec,)
    raise fail(f"unknown tuple: {selection}")


def run_lock(root: Path, specs: tuple[TupleSpec, ...], cutoff: str) -> None:
    for spec in specs:
        command = [
            "uv",
            "pip",
            "compile",
            spec.requirements_input,
            "--output-file",
            spec.requirements_lock,
            "--generate-hashes",
            "--quiet",
            "--torch-backend",
            spec.torch_backend,
            "--exclude-newer",
            cutoff,
        ]
        # Registry values are validated before command assembly.
        subprocess.run(command, check=True, cwd=root)  # nosec B603


def docker_build(root: Path, specs: tuple[TupleSpec, ...]) -> None:
    for spec in specs:
        command = [
            "docker",
            "build",
            "--file",
            DEFAULT_DOCKERFILE,
            "--tag",
            spec.image_tag,
        ]
        for name, value in spec.build_args().items():
            command.extend(["--build-arg", f"{name}={value}"])
        command.append(".")
        # Registry values are validated before command assembly.
        subprocess.run(command, check=True, cwd=root)  # nosec B603


def docker_verify(root: Path, specs: tuple[TupleSpec, ...], gpu: bool) -> None:
    if gpu:
        specs = tuple(spec for spec in specs if spec.is_cuda)
        if not specs:
            raise fail("GPU verification needs at least one CUDA tuple")
    for spec in specs:
        gpu_arguments = ["--gpus", "all"] if gpu else []
        gpu_assertion = (
            "assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
            if gpu
            else ""
        )
        check = (
            "import os, sys, torch; "
            "assert (os.getuid(), os.getgid()) == (1000, 1000); "
            f"assert f'{{sys.version_info.major}}.{{sys.version_info.minor}}' == {spec.python_version!r}; "
            f"assert torch.__version__ == {spec.torch_version!r}; "
            f"assert (torch.version.cuda or 'none') == {spec.cuda_version!r}; "
            f"{gpu_assertion}"
        )
        command = [
            "docker",
            "run",
            "--rm",
            *gpu_arguments,
            "--read-only",
            "--tmpfs",
            CONTAINER_TMPFS,
            spec.image_tag,
            "python",
            "-c",
            check,
        ]
        # Registry values are validated before command assembly.
        subprocess.run(command, check=True, cwd=root)  # nosec B603


def render_ci_targets(specs: tuple[TupleSpec, ...]) -> str:
    targets = []
    for spec in specs:
        targets.append(
            {
                "file": DEFAULT_DOCKERFILE,
                "tag_prefix": spec.tag_prefix,
                "tag_suffix": spec.tag_suffix,
                "platforms": spec.platforms,
                "build_args": "\n".join(
                    f"{name}={value}" for name, value in spec.build_args().items()
                ),
            }
        )
    return json.dumps(targets, separators=(",", ":"), sort_keys=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="tuple registry path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the tuple registry")
    subparsers.add_parser("list", help="list registered tuple ids")
    subparsers.add_parser(
        "ci-targets", help="emit reusable Docker workflow build targets"
    )
    lock = subparsers.add_parser(
        "lock", help="generate one or more hash-locked requirements files"
    )
    lock.add_argument(
        "--cutoff", required=True, help="dependency publication cutoff timestamp"
    )
    lock.add_argument("selection", help="tuple id, cpu, cuda, or all")
    for command in ("docker-build", "docker-verify"):
        operation = subparsers.add_parser(
            command, help=f"{command.replace('-', ' ')} one or more tuple images"
        )
        if command == "docker-verify":
            operation.add_argument(
                "--gpu", action="store_true", help="require a CUDA-capable GPU"
            )
        operation.add_argument("selection", help="tuple id, cpu, cuda, or all")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        root, specs = load_specs(arguments.registry.resolve())
        if arguments.command == "validate":
            print(f"validated {len(specs)} tuple(s)")
            return 0
        if arguments.command == "list":
            print("\n".join(spec.identifier for spec in specs))
            return 0
        if arguments.command == "ci-targets":
            print(render_ci_targets(specs))
            return 0
        selected = select_specs(specs, arguments.selection)
        if not selected:
            raise fail(f"selection has no matching tuples: {arguments.selection}")
        if arguments.command == "lock":
            run_lock(root, selected, arguments.cutoff)
        elif arguments.command == "docker-build":
            docker_build(root, selected)
        elif arguments.command == "docker-verify":
            docker_verify(root, selected, arguments.gpu)
    except RegistryError as error:
        print(f"tuple matrix error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
