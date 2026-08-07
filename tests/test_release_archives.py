from __future__ import annotations

import copy
import io
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UV = PROJECT_ROOT / ".tools" / "uv.exe"
VERIFIER = PROJECT_ROOT / "scripts" / "verify_release_archives.py"
WHEEL_NAME = "adaptive_edge_perception-0.1.0-py3-none-any.whl"
SDIST_NAME = "adaptive_edge_perception-0.1.0.tar.gz"
DIST_INFO = "adaptive_edge_perception-0.1.0.dist-info"
SDIST_ROOT = "adaptive_edge_perception-0.1.0"


@pytest.fixture(scope="module")
def release_archives(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    dist = tmp_path_factory.mktemp("release-archives")
    result = subprocess.run(
        [str(UV), "build", "--offline", "--out-dir", str(dist)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return dist / WHEEL_NAME, dist / SDIST_NAME


def _rewrite_wheel(
    source: Path,
    destination: Path,
    *,
    omitted_member: str | None = None,
    replacements: dict[str, bytes] | None = None,
    extra_member: str | None = None,
) -> None:
    replacement_map = replacements or {}
    with zipfile.ZipFile(source) as input_archive, zipfile.ZipFile(destination, "w") as output_archive:
        for info in input_archive.infolist():
            if info.filename == omitted_member:
                continue
            output_archive.writestr(
                info, replacement_map.get(info.filename, input_archive.read(info))
            )
        if extra_member is not None:
            output_archive.writestr(extra_member, b"blocked checkout state")


def _rewrite_sdist(
    source: Path,
    destination: Path,
    *,
    omitted_member: str | None = None,
    replacements: dict[str, bytes] | None = None,
    extra_member: str | None = None,
) -> None:
    replacement_map = replacements or {}
    with tarfile.open(source, mode="r:gz") as input_archive, tarfile.open(
        destination, mode="w:gz"
    ) as output_archive:
        for member in input_archive.getmembers():
            if member.name == omitted_member:
                continue
            data_file = input_archive.extractfile(member)
            if data_file is None:
                output_archive.addfile(member)
                continue
            data = replacement_map.get(member.name, data_file.read())
            copied_member = copy.copy(member)
            copied_member.size = len(data)
            output_archive.addfile(copied_member, io.BytesIO(data))
        if extra_member is not None:
            data = b"blocked checkout state"
            info = tarfile.TarInfo(extra_member)
            info.size = len(data)
            info.mtime = 0
            output_archive.addfile(info, io.BytesIO(data))


def _run_verifier(wheel: Path, sdist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFIER), str(wheel), str(sdist)],
        check=False,
        capture_output=True,
        text=True,
    )


def _mutated_wheel(release_archives: tuple[Path, Path], destination: Path, **kwargs: object) -> Path:
    _rewrite_wheel(release_archives[0], destination, **kwargs)
    return destination


def _mutated_sdist(release_archives: tuple[Path, Path], destination: Path, **kwargs: object) -> Path:
    _rewrite_sdist(release_archives[1], destination, **kwargs)
    return destination


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


def test_release_verifier_accepts_freshly_built_release_contract(
    release_archives: tuple[Path, Path],
) -> None:
    result = _run_verifier(*release_archives)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines()[-1] == "release archive policy: passed"


@pytest.mark.parametrize(
    ("archive_kind", "member_name", "error_member"),
    [
        ("wheel", "edge_perception/worker.py", "edge_perception/worker.py"),
        ("sdist", f"{SDIST_ROOT}/src/edge_perception/worker.py", "src/edge_perception/worker.py"),
        ("sdist", f"{SDIST_ROOT}/tests/test_worker.py", "tests/test_worker.py"),
        (
            "sdist",
            f"{SDIST_ROOT}/scripts/verify_release_archives.py",
            "scripts/verify_release_archives.py",
        ),
        (
            "sdist",
            f"{SDIST_ROOT}/docs/checkpoints/eyes-and-stopwatch.md",
            "docs/checkpoints/eyes-and-stopwatch.md",
        ),
        ("sdist", f"{SDIST_ROOT}/LICENSE", "LICENSE"),
    ],
)
def test_release_verifier_rejects_each_removed_required_member(
    release_archives: tuple[Path, Path],
    tmp_path: Path,
    archive_kind: str,
    member_name: str,
    error_member: str,
) -> None:
    wheel, sdist = release_archives
    if archive_kind == "wheel":
        wheel = _mutated_wheel(release_archives, tmp_path / WHEEL_NAME, omitted_member=member_name)
    else:
        sdist = _mutated_sdist(release_archives, tmp_path / SDIST_NAME, omitted_member=member_name)

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"error: {archive_kind} is missing required archive member: {error_member}\n"


@pytest.mark.parametrize(
    ("archive_kind", "member_name"),
    [
        ("wheel", f"{DIST_INFO}/licenses/LICENSE"),
        ("sdist", f"{SDIST_ROOT}/LICENSE"),
    ],
)
def test_release_verifier_rejects_replaced_license_bytes(
    release_archives: tuple[Path, Path],
    tmp_path: Path,
    archive_kind: str,
    member_name: str,
) -> None:
    wheel, sdist = release_archives
    if archive_kind == "wheel":
        wheel = _mutated_wheel(
            release_archives,
            tmp_path / WHEEL_NAME,
            replacements={member_name: b"replaced wheel license\n"},
        )
    else:
        sdist = _mutated_sdist(
            release_archives,
            tmp_path / SDIST_NAME,
            replacements={member_name: b"replaced sdist license\n"},
        )

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"error: {archive_kind} LICENSE does not match repository LICENSE\n"


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_release_verifier_rejects_blocked_checkout_state(
    release_archives: tuple[Path, Path], tmp_path: Path, archive_kind: str
) -> None:
    wheel, sdist = release_archives
    if archive_kind == "wheel":
        wheel = _mutated_wheel(
            release_archives,
            tmp_path / WHEEL_NAME,
            extra_member="edge_perception/private/model.safetensors",
        )
        expected_member = "edge_perception/private/model.safetensors"
    else:
        sdist = _mutated_sdist(
            release_archives,
            tmp_path / SDIST_NAME,
            extra_member=f"{SDIST_ROOT}/.superpowers/brainstorm/.last-token",
        )
        expected_member = ".superpowers/brainstorm/.last-token"

    result = _run_verifier(wheel, sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        f"error: {archive_kind} contains unexpected archive member: {expected_member}\n"
    )


def test_release_verifier_rejects_missing_wheel_pep639_license_metadata(
    release_archives: tuple[Path, Path], tmp_path: Path
) -> None:
    wheel = _mutated_wheel(
        release_archives,
        tmp_path / WHEEL_NAME,
        replacements={f"{DIST_INFO}/METADATA": b"Metadata-Version: 2.4\n"},
    )

    result = _run_verifier(wheel, release_archives[1])

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "error: wheel metadata License-Expression must be Apache-2.0\n"


def test_release_verifier_rejects_missing_sdist_pep639_license_metadata(
    release_archives: tuple[Path, Path], tmp_path: Path
) -> None:
    sdist = _mutated_sdist(
        release_archives,
        tmp_path / SDIST_NAME,
        replacements={f"{SDIST_ROOT}/PKG-INFO": b"Metadata-Version: 2.4\n"},
    )

    result = _run_verifier(release_archives[0], sdist)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "error: sdist metadata License-Expression must be Apache-2.0\n"
