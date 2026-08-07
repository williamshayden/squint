#!/usr/bin/env python3
"""Verify the exact public file and license-metadata contract for release archives.

Set ``RELEASE_ARCHIVES_GIT`` only when Git is not discoverable on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Sequence
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_NAME = "adaptive-edge-perception"
VERSION = "0.1.0"
WHEEL_NAME = f"adaptive_edge_perception-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"adaptive_edge_perception-{VERSION}.tar.gz"
DIST_INFO = f"adaptive_edge_perception-{VERSION}.dist-info"
SDIST_ROOT = f"adaptive_edge_perception-{VERSION}"

WHEEL_METADATA_MEMBERS = frozenset(
    {
        f"{DIST_INFO}/METADATA",
        f"{DIST_INFO}/WHEEL",
        f"{DIST_INFO}/entry_points.txt",
        f"{DIST_INFO}/licenses/LICENSE",
        f"{DIST_INFO}/RECORD",
    }
)
SDIST_PROJECT_MEMBERS = frozenset(
    {"LICENSE", "README.md", "pyproject.toml", "uv.lock", "PKG-INFO"}
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT_ENV_VAR = "RELEASE_ARCHIVES_GIT"


try:
    from hatchling.builders.hooks.plugin.interface import (  # type: ignore[import-not-found]
        BuildHookInterface,
    )
except ModuleNotFoundError:
    BuildHookInterface = None

if __name__ != "__main__" and BuildHookInterface is not None:
    class CustomBuildHook(BuildHookInterface):  # type: ignore[misc]
        """Keep Hatch's VCS filters without publishing the filter files themselves."""

        def initialize(self, version: str, build_data: dict[str, Any]) -> None:
            force_include = build_data.get("force_include")
            if not isinstance(force_include, dict):
                return
            for source in tuple(force_include):
                if Path(source).name in {".gitignore", ".hgignore"}:
                    force_include.pop(source)


class ArchivePolicyError(ValueError):
    """A release archive violates the public release policy."""


def _resolve_git() -> str:
    override = os.environ.get(GIT_ENV_VAR)
    if override:
        return override
    git = shutil.which("git")
    if git:
        return git
    raise ArchivePolicyError("cannot find Git on PATH")


def _git_head_bytes(*arguments: str) -> bytes:
    try:
        result = subprocess.run(
            [_resolve_git(), "-C", str(PROJECT_ROOT), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ArchivePolicyError("cannot read the clean project contract from Git") from error
    if result.returncode != 0:
        raise ArchivePolicyError("cannot read the clean project contract from Git")
    return result.stdout


def _clean_project_paths() -> frozenset[str]:
    output = _git_head_bytes("ls-tree", "-r", "-z", "--name-only", "HEAD")
    return frozenset(path for path in output.decode("utf-8").split("\0") if path)


def _expected_wheel_members(project_paths: frozenset[str]) -> frozenset[str]:
    package_members = {
        path.removeprefix("src/")
        for path in project_paths
        if _is_python_member(path, "src/edge_perception/")
    }
    return frozenset(package_members).union(WHEEL_METADATA_MEMBERS)


def _expected_sdist_members(project_paths: frozenset[str]) -> frozenset[str]:
    source_members = {
        path
        for path in project_paths
        if _is_python_member(path, "src/edge_perception/")
        or _is_python_member(path, "tests/")
        or path == "scripts/verify_release_archives.py"
        or (
            path.startswith("docs/checkpoints/")
            and path.endswith(".md")
            and len(path) > len("docs/checkpoints/")
        )
    }
    return frozenset(source_members).union(SDIST_PROJECT_MEMBERS)


def _validate_exact_members(
    actual_members: set[str], expected_members: frozenset[str], archive_kind: str
) -> None:
    unexpected = sorted(actual_members.difference(expected_members))
    if unexpected:
        raise ArchivePolicyError(
            f"{archive_kind} contains unexpected archive member: {unexpected[0]}"
        )
    missing = sorted(expected_members.difference(actual_members))
    if missing:
        raise ArchivePolicyError(
            f"{archive_kind} is missing required archive member: {missing[0]}"
        )


def _clean_license_bytes() -> bytes:
    return _git_head_bytes("show", "HEAD:LICENSE")


def _validate_member_name(name: str, archive_kind: str) -> None:
    raw_parts = name.split("/")
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ArchivePolicyError(
            f"{archive_kind} contains unexpected archive member: {name}"
        )


def _is_python_member(name: str, prefix: str) -> bool:
    return name.startswith(prefix) and name.endswith(".py") and len(name) > len(prefix)


def _validate_metadata(raw_metadata: bytes, archive_kind: str) -> None:
    metadata = BytesParser(policy=policy.compat32).parsebytes(raw_metadata)
    if metadata.get_all("License-Expression", []) != ["Apache-2.0"]:
        raise ArchivePolicyError(
            f"{archive_kind} metadata License-Expression must be Apache-2.0"
        )
    if metadata.get_all("License-File", []) != ["LICENSE"]:
        raise ArchivePolicyError(
            f"{archive_kind} metadata License-File must be LICENSE"
        )
    if metadata.get("Metadata-Version") != "2.4":
        raise ArchivePolicyError(
            f"{archive_kind} metadata Metadata-Version must be 2.4"
        )
    if metadata.get("Name") != PROJECT_NAME:
        raise ArchivePolicyError(
            f"{archive_kind} metadata Name must be {PROJECT_NAME}"
        )
    if metadata.get("Version") != VERSION:
        raise ArchivePolicyError(
            f"{archive_kind} metadata Version must be {VERSION}"
        )


def _verify_wheel(path: Path) -> str:
    if path.name != WHEEL_NAME:
        raise ArchivePolicyError(f"wheel filename must be {WHEEL_NAME}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ArchivePolicyError("wheel contains duplicate archive members")

        for info in sorted(infos, key=lambda candidate: candidate.filename):
            name = info.filename
            _validate_member_name(name, "wheel")
            if info.is_dir():
                raise ArchivePolicyError(
                    f"wheel contains unexpected archive member: {name}"
                )
        _validate_exact_members(
            set(names), _expected_wheel_members(_clean_project_paths()), "wheel"
        )
        _validate_metadata(archive.read(f"{DIST_INFO}/METADATA"), "wheel")
        if archive.read(f"{DIST_INFO}/licenses/LICENSE") != _clean_license_bytes():
            raise ArchivePolicyError("wheel LICENSE does not match repository LICENSE")

    return (
        f"wheel inventory: {len(names)} files; "
        f"{sum(_is_python_member(name, 'edge_perception/') for name in names)} package Python; "
        f"{len(WHEEL_METADATA_MEMBERS)} metadata/license"
    )


def _sdist_relative_name(name: str) -> str:
    _validate_member_name(name, "sdist")
    root, separator, relative = name.partition("/")
    if root != SDIST_ROOT or not separator or not relative:
        raise ArchivePolicyError(f"sdist contains unexpected archive member: {name}")
    _validate_member_name(relative, "sdist")
    return relative


def _verify_sdist(path: Path) -> str:
    if path.name != SDIST_NAME:
        raise ArchivePolicyError(f"sdist filename must be {SDIST_NAME}")
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        relative_members: dict[str, tarfile.TarInfo] = {}
        for member in members:
            relative = _sdist_relative_name(member.name)
            if relative in relative_members:
                raise ArchivePolicyError("sdist contains duplicate archive members")
            if not member.isfile():
                raise ArchivePolicyError(
                    f"sdist contains unexpected archive member: {relative}"
                )
            relative_members[relative] = member

        _validate_exact_members(
            set(relative_members), _expected_sdist_members(_clean_project_paths()), "sdist"
        )

        metadata_member = relative_members["PKG-INFO"]
        metadata_file = archive.extractfile(metadata_member)
        if metadata_file is None:
            raise ArchivePolicyError("sdist cannot read required archive member: PKG-INFO")
        _validate_metadata(metadata_file.read(), "sdist")
        license_file = archive.extractfile(relative_members["LICENSE"])
        if license_file is None:
            raise ArchivePolicyError("sdist cannot read required archive member: LICENSE")
        if license_file.read() != _clean_license_bytes():
            raise ArchivePolicyError("sdist LICENSE does not match repository LICENSE")

    return (
        f"sdist inventory: {len(relative_members)} files; "
        f"{sum(_is_python_member(name, 'src/edge_perception/') for name in relative_members)} "
        f"package Python; {sum(_is_python_member(name, 'tests/') for name in relative_members)} "
        f"test Python; 1 release verifier; "
        f"{sum(name.startswith('docs/checkpoints/') for name in relative_members)} checkpoint; "
        f"{len(SDIST_PROJECT_MEMBERS)} project/metadata"
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("error: expected WHEEL and SDIST paths", file=sys.stderr)
        return 1
    try:
        wheel_inventory = _verify_wheel(Path(arguments[0]))
        sdist_inventory = _verify_sdist(Path(arguments[1]))
    except (ArchivePolicyError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(wheel_inventory)
    print(sdist_inventory)
    print("release archive policy: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
