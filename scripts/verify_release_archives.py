#!/usr/bin/env python3
"""Verify the exact public file and license-metadata contract for release archives."""

from __future__ import annotations

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


if __name__ != "__main__":
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    class CustomBuildHook(BuildHookInterface):
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

        package_python = 0
        metadata_members = 0
        for info in sorted(infos, key=lambda candidate: candidate.filename):
            name = info.filename
            _validate_member_name(name, "wheel")
            if info.is_dir():
                raise ArchivePolicyError(
                    f"wheel contains unexpected archive member: {name}"
                )
            if _is_python_member(name, "edge_perception/"):
                package_python += 1
            elif name in WHEEL_METADATA_MEMBERS:
                metadata_members += 1
            else:
                raise ArchivePolicyError(
                    f"wheel contains unexpected archive member: {name}"
                )

        missing = sorted(WHEEL_METADATA_MEMBERS.difference(names))
        if missing:
            raise ArchivePolicyError(f"wheel is missing required archive member: {missing[0]}")
        if "edge_perception/__init__.py" not in names:
            raise ArchivePolicyError(
                "wheel is missing required archive member: edge_perception/__init__.py"
            )
        _validate_metadata(archive.read(f"{DIST_INFO}/METADATA"), "wheel")

    return (
        f"wheel inventory: {len(names)} files; {package_python} package Python; "
        f"{metadata_members} metadata/license"
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

        package_python = 0
        test_python = 0
        release_verifiers = 0
        checkpoints = 0
        project_metadata = 0
        for relative in sorted(relative_members):
            if relative in SDIST_PROJECT_MEMBERS:
                project_metadata += 1
            elif _is_python_member(relative, "src/edge_perception/"):
                package_python += 1
            elif _is_python_member(relative, "tests/"):
                test_python += 1
            elif relative == "scripts/verify_release_archives.py":
                release_verifiers += 1
            elif (
                relative.startswith("docs/checkpoints/")
                and relative.endswith(".md")
                and len(relative) > len("docs/checkpoints/")
            ):
                checkpoints += 1
            else:
                raise ArchivePolicyError(
                    f"sdist contains unexpected archive member: {relative}"
                )

        missing = sorted(SDIST_PROJECT_MEMBERS.difference(relative_members))
        if missing:
            raise ArchivePolicyError(f"sdist is missing required archive member: {missing[0]}")
        required_family_counts = (
            (package_python, "src/edge_perception package Python"),
            (test_python, "tests Python"),
            (release_verifiers, "scripts/verify_release_archives.py"),
            (checkpoints, "docs/checkpoints Markdown"),
        )
        for count, family in required_family_counts:
            if count == 0:
                raise ArchivePolicyError(f"sdist is missing required member family: {family}")

        metadata_member = relative_members["PKG-INFO"]
        metadata_file = archive.extractfile(metadata_member)
        if metadata_file is None:
            raise ArchivePolicyError("sdist cannot read required archive member: PKG-INFO")
        _validate_metadata(metadata_file.read(), "sdist")

    return (
        f"sdist inventory: {len(relative_members)} files; "
        f"{package_python} package Python; {test_python} test Python; "
        f"{release_verifiers} release verifier; {checkpoints} checkpoint; "
        f"{project_metadata} project/metadata"
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
