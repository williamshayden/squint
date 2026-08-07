from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = PROJECT_ROOT / "scripts" / "verify_release_archives.py"
SDIST_ROOT = "adaptive_edge_perception-0.1.0"
WHEEL_METADATA = (
    b"Metadata-Version: 2.4\n"
    b"Name: adaptive-edge-perception\n"
    b"Version: 0.1.0\n"
    b"License-Expression: Apache-2.0\n"
    b"License-File: LICENSE\n"
)
SDIST_METADATA = WHEEL_METADATA


def _write_wheel(
    path: Path,
    *,
    extra_member: str | None = None,
    metadata: bytes = WHEEL_METADATA,
) -> None:
    members = {
        "edge_perception/__init__.py": b"",
        "adaptive_edge_perception-0.1.0.dist-info/METADATA": metadata,
        "adaptive_edge_perception-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        "adaptive_edge_perception-0.1.0.dist-info/entry_points.txt": b"",
        "adaptive_edge_perception-0.1.0.dist-info/licenses/LICENSE": b"Apache License\n",
        "adaptive_edge_perception-0.1.0.dist-info/RECORD": b"",
    }
    if extra_member is not None:
        members[extra_member] = b"unexpected"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _write_sdist(
    path: Path,
    *,
    extra_member: str | None = None,
    metadata: bytes = SDIST_METADATA,
) -> None:
    members = {
        "LICENSE": b"Apache License\n",
        "README.md": b"# Project\n",
        "pyproject.toml": b"[project]\nname = 'adaptive-edge-perception'\n",
        "uv.lock": b"version = 1\n",
        "PKG-INFO": metadata,
        "src/edge_perception/__init__.py": b"",
        "tests/test_smoke.py": b"def test_smoke(): pass\n",
        "scripts/verify_release_archives.py": b"",
        "docs/checkpoints/eyes-and-stopwatch.md": b"# Checkpoint\n",
    }
    if extra_member is not None:
        members[extra_member] = b"unexpected"
    with tarfile.open(path, "w:gz") as archive:
        for relative_name, content in members.items():
            info = tarfile.TarInfo(f"{SDIST_ROOT}/{relative_name}")
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))


def _run_verifier(wheel: Path, sdist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFIER), str(wheel), str(sdist)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_project_declares_pep639_license_and_bounded_hatchling() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["build-system"]["requires"] == ["hatchling>=1.27,<2"]
    assert project["project"]["license"] == "Apache-2.0"
    assert project["project"]["license-files"] == ["LICENSE"]
    targets = project["tool"]["hatch"]["build"]["targets"]
    assert targets["wheel"] == {
        "only-include": ["src/edge_perception"],
        "sources": ["src"],
    }
    assert targets["sdist"] == {
        "only-include": [
            "LICENSE",
            "README.md",
            "pyproject.toml",
            "uv.lock",
            "src",
            "tests",
            "scripts/verify_release_archives.py",
            "docs/checkpoints",
        ]
    }
    assert project["tool"]["hatch"]["build"]["hooks"]["custom"] == {
        "path": "scripts/verify_release_archives.py"
    }
    assert (PROJECT_ROOT / "LICENSE").is_file()


def test_release_verifier_accepts_only_the_public_wheel_and_sdist_contract(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "adaptive_edge_perception-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "adaptive_edge_perception-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "wheel inventory: 6 files; 1 package Python; 5 metadata/license",
        (
            "sdist inventory: 9 files; 1 package Python; 1 test Python; "
            "1 release verifier; 1 checkpoint; 5 project/metadata"
        ),
        "release archive policy: passed",
    ]


def test_release_verifier_rejects_ignored_checkout_state_from_wheel(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "adaptive_edge_perception-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "adaptive_edge_perception-0.1.0.tar.gz"
    _write_wheel(wheel, extra_member="edge_perception/private/model.safetensors")
    _write_sdist(sdist)

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "error: wheel contains unexpected archive member: "
        "edge_perception/private/model.safetensors\n"
    )


def test_release_verifier_rejects_ignored_checkout_state_from_sdist(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "adaptive_edge_perception-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "adaptive_edge_perception-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist, extra_member=".superpowers/brainstorm/.last-token")

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "error: sdist contains unexpected archive member: "
        ".superpowers/brainstorm/.last-token\n"
    )


def test_release_verifier_rejects_missing_pep639_license_metadata(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "adaptive_edge_perception-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "adaptive_edge_perception-0.1.0.tar.gz"
    _write_wheel(wheel, metadata=b"Metadata-Version: 2.4\n")
    _write_sdist(sdist)

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "error: wheel metadata License-Expression must be Apache-2.0\n"
    )


def test_release_verifier_checks_sdist_pep639_license_metadata(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "adaptive_edge_perception-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "adaptive_edge_perception-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist, metadata=b"Metadata-Version: 2.4\n")

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "error: sdist metadata License-Expression must be Apache-2.0\n"
    )
