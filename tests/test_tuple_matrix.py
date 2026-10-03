from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "tuple_matrix.py"
SHA256_DIGEST = "a" * 64
EXPECTED_TUPLE_IDS = {
    "py3.12-torch2.5.1-cpu",
    "py3.12-torch2.5.1-cu124",
    "py3.12-torch2.14-cpu",
    "py3.12-torch2.14-cu126",
    "py3.12-torch2.14-cu130",
    "py3.13-torch2.14-cpu",
    "py3.13-torch2.14-cu126",
    "py3.13-torch2.14-cu130",
}


def run_matrix(
    *arguments: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    command_environment = os.environ.copy()
    if environment:
        command_environment.update(environment)
    command = [sys.executable, str(SCRIPT), *arguments]
    if command_environment.get("COVERAGE_RUN") == "1":
        command = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--branch",
            str(SCRIPT),
            *arguments,
        ]
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        env=command_environment,
        text=True,
    )


def write_command_stub(directory: Path, name: str) -> Path:
    command_path = directory / name
    command_path.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" >> "$COMMAND_LOG_PATH"\n',
        encoding="utf-8",
    )
    command_path.chmod(0o755)
    return command_path


def valid_tuple() -> dict[str, object]:
    return {
        "id": "safe-id",
        "tag_prefix": "safe-",
        "tag_suffix": "-cpu",
        "local_tag": "safe",
        "platforms": "linux/amd64",
        "python_version": "3.12",
        "python_builder_image": f"python:3.12@sha256:{SHA256_DIGEST}",
        "runtime_image": f"python:3.12@sha256:{SHA256_DIGEST}",
        "runtime_apt_packages": [],
        "runtime_python": "",
        "requirements_input": "requirements.in",
        "requirements_lock": "requirements.txt",
        "torch_backend": "cpu",
        "torch_version": "2.14.0+cpu",
        "cuda_version": "none",
    }


def write_registry(directory: Path, document: dict[str, object]) -> Path:
    (directory / "requirements.in").write_text("torch==2.14.0\n", encoding="utf-8")
    (directory / "requirements.txt").write_text("torch==2.14.0\n", encoding="utf-8")
    registry_path = directory / "tuples.json"
    registry_path.write_text(json.dumps(document), encoding="utf-8")
    return registry_path


class TupleMatrixCommandTests(unittest.TestCase):
    def test_ci_targets_describe_each_registered_tuple(self) -> None:
        result = run_matrix("ci-targets")

        self.assertEqual(result.returncode, 0, result.stderr)
        targets = json.loads(result.stdout)

        self.assertEqual(len(targets), len(EXPECTED_TUPLE_IDS))
        image_shapes = {
            f"{target['tag_prefix']}latest{target['tag_suffix']}" for target in targets
        }
        self.assertEqual(
            image_shapes,
            {
                "py3.12-torch2.5.1-latest-cpu",
                "py3.12-torch2.5.1-latest-cu124",
                "py3.12-torch2.14-latest-cpu",
                "py3.12-torch2.14-latest-cu126",
                "py3.12-torch2.14-latest-cu130",
                "py3.13-torch2.14-latest-cpu",
                "py3.13-torch2.14-latest-cu126",
                "py3.13-torch2.14-latest-cu130",
            },
        )
        self.assertTrue(all(target["file"] == "Dockerfile" for target in targets))
        self.assertTrue(
            all("PYTHON_BUILDER_IMAGE=" in target["build_args"] for target in targets)
        )
        self.assertTrue(
            any(
                "EXPECTED_PYTHON_VERSION=3.12" in target["build_args"]
                for target in targets
            )
        )
        self.assertTrue(
            any(
                "EXPECTED_PYTHON_VERSION=3.13" in target["build_args"]
                for target in targets
            )
        )

    def test_list_exposes_every_supported_tuple(self) -> None:
        result = run_matrix("list")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), EXPECTED_TUPLE_IDS)

    def test_audiolla_tuples_select_matching_torch_and_backend(self) -> None:
        for backend, expected_cuda in (("cpu", "none"), ("cu124", "12.4")):
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                temporary_root = Path(temporary_directory)
                command_log = temporary_root / "docker-command.log"
                write_command_stub(temporary_root, "docker")
                result = run_matrix(
                    "docker-build",
                    f"py3.12-torch2.5.1-{backend}",
                    environment={
                        "COMMAND_LOG_PATH": str(command_log),
                        "PATH": f"{temporary_root}:{os.environ['PATH']}",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                invocation = command_log.read_text(encoding="utf-8")
                self.assertEqual(invocation.splitlines().count("build"), 1)
                self.assertIn(f"TORCH_BACKEND={backend}", invocation)
                self.assertIn(f"EXPECTED_TORCH_VERSION=2.5.1+{backend}", invocation)
                self.assertIn(f"EXPECTED_CUDA_VERSION={expected_cuda}", invocation)
                self.assertIn(
                    f"REQUIREMENTS_FILE=requirements-py312-torch251-{backend}.txt",
                    invocation,
                )

    def test_unknown_tuple_fails_before_a_docker_command_can_run(self) -> None:
        result = run_matrix("docker-build", "does-not-exist")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("unknown tuple: does-not-exist", result.stderr)

    def test_unsafe_tag_suffix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            tuple_document = valid_tuple()
            tuple_document["tag_suffix"] = ";touch-owned-file"
            registry_path = write_registry(temporary_root, {"tuples": [tuple_document]})

            result = run_matrix("--registry", str(registry_path), "validate")

        self.assertEqual(result.returncode, 2)
        self.assertIn("tag_suffix is not a safe image tag part", result.stderr)

    def test_invalid_registry_values_are_rejected_before_external_commands(
        self,
    ) -> None:
        invalid_values = {
            "id": "not safe",
            "local_tag": "not safe",
            "python_builder_image": "python:3.12",
            "runtime_apt_packages": ["gcc;touch-owned-file"],
            "requirements_input": "../requirements.in",
            "platforms": "windows/amd64",
            "python_version": "3",
            "runtime_python": "usr/bin/python",
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    tuple_document = valid_tuple()
                    tuple_document[field] = value
                    registry_path = write_registry(
                        temporary_root, {"tuples": [tuple_document]}
                    )

                    result = run_matrix("--registry", str(registry_path), "validate")

                self.assertEqual(result.returncode, 2)
                self.assertIn("tuple matrix error:", result.stderr)

    def test_invalid_cuda_combinations_are_rejected(self) -> None:
        invalid_combinations = (
            {"cuda_version": "12.6"},
            {"torch_backend": "cuda", "cuda_version": "12.6"},
        )
        for changes in invalid_combinations:
            with self.subTest(changes=changes):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    tuple_document = valid_tuple()
                    tuple_document.update(changes)
                    registry_path = write_registry(
                        temporary_root, {"tuples": [tuple_document]}
                    )

                    result = run_matrix("--registry", str(registry_path), "validate")

                self.assertEqual(result.returncode, 2)
                self.assertIn("tuple matrix error:", result.stderr)

    def test_invalid_registry_documents_are_rejected(self) -> None:
        documents = (
            "{",
            json.dumps({"tuples": []}),
            json.dumps({"tuples": [valid_tuple(), valid_tuple()]}),
        )
        for document in documents:
            with self.subTest(document=document):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    registry_path = Path(temporary_directory) / "tuples.json"
                    registry_path.write_text(document, encoding="utf-8")

                    result = run_matrix("--registry", str(registry_path), "validate")

                self.assertEqual(result.returncode, 2)
                self.assertIn("tuple matrix error:", result.stderr)

    def test_missing_registry_or_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            missing_result = run_matrix(
                "--registry", str(temporary_root / "missing.json"), "validate"
            )
            registry_path = write_registry(temporary_root, {"tuples": [valid_tuple()]})
            (temporary_root / "requirements.txt").unlink()
            lock_result = run_matrix("--registry", str(registry_path), "validate")

        self.assertEqual(missing_result.returncode, 2)
        self.assertIn("tuple registry not found", missing_result.stderr)
        self.assertEqual(lock_result.returncode, 2)
        self.assertIn("requirements lock missing", lock_result.stderr)

    def test_lock_passes_the_selected_backend_to_uv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            command_log = temporary_root / "uv-command.log"
            write_command_stub(temporary_root, "uv")

            result = run_matrix(
                "lock",
                "cpu",
                "--cutoff",
                "2026-09-20T00:00:00Z",
                environment={
                    "COMMAND_LOG_PATH": str(command_log),
                    "PATH": f"{temporary_root}:{os.environ['PATH']}",
                },
            )
            invocation = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pip\ncompile\nrequirements-cpu.in", invocation)
        self.assertIn("pip\ncompile\nrequirements-py313-cpu.in", invocation)
        self.assertIn("--torch-backend\ncpu", invocation)

    def test_docker_build_uses_each_selected_cpu_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            command_log = temporary_root / "docker-command.log"
            write_command_stub(temporary_root, "docker")

            result = run_matrix(
                "docker-build",
                "cpu",
                environment={
                    "COMMAND_LOG_PATH": str(command_log),
                    "PATH": f"{temporary_root}:{os.environ['PATH']}",
                },
            )
            invocation = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("build\n--file\nDockerfile", invocation)
        self.assertIn("--tag\npsyb0t/torchbase:local-py312-cpu", invocation)
        self.assertIn("--tag\npsyb0t/torchbase:local-py313-cpu", invocation)
        self.assertIn("TORCH_BACKEND=cpu", invocation)

    def test_gpu_verification_requests_a_gpu_for_cuda_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            command_log = temporary_root / "docker-command.log"
            write_command_stub(temporary_root, "docker")

            result = run_matrix(
                "docker-verify",
                "--gpu",
                "cuda",
                environment={
                    "COMMAND_LOG_PATH": str(command_log),
                    "PATH": f"{temporary_root}:{os.environ['PATH']}",
                },
            )
            invocation = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run\n--rm\n--gpus\nall", invocation)
        for tuple_id in (
            "local-py312-cu126",
            "local-py312-cu130",
            "local-py313-cu126",
            "local-py313-cu130",
        ):
            self.assertIn(f"psyb0t/torchbase:{tuple_id}", invocation)

    def test_gpu_verification_rejects_cpu_only_selection(self) -> None:
        result = run_matrix("docker-verify", "--gpu", "cpu")

        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU verification needs at least one CUDA tuple", result.stderr)

    def test_external_command_failure_is_returned_to_the_caller(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            command_path = write_command_stub(temporary_root, "docker")
            command_path.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")

            result = run_matrix(
                "docker-build",
                "py3.13-torch2.14-cpu",
                environment={"PATH": f"{temporary_root}:{os.environ['PATH']}"},
            )

        self.assertEqual(result.returncode, 23)


if __name__ == "__main__":
    unittest.main()
