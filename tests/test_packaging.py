from importlib import import_module
from importlib.metadata import PackageNotFoundError, entry_points, metadata


def _project_metadata():
    try:
        return metadata("squint-rl")
    except PackageNotFoundError:
        return None


def test_distribution_and_import_names_are_distinct() -> None:
    project = _project_metadata()
    assert project is not None, "squint-rl distribution metadata is missing"
    squint_rl = import_module("squint_rl")
    assert project["Name"] == "squint-rl"
    assert squint_rl.__version__ == project["Version"] == "0.1.0"


def test_core_metadata_keeps_heavy_dependencies_optional() -> None:
    project = _project_metadata()
    assert project is not None, "squint-rl distribution metadata is missing"
    requirements = project.get_all("Requires-Dist") or []
    unconditional = [item.lower() for item in requirements if "extra ==" not in item.lower()]
    assert not any(
        name in item
        for name in ("torch", "transformers", "trackers", "pyside")
        for item in unconditional
    )


def test_console_entry_point_and_version(capsys) -> None:
    project = _project_metadata()
    assert project is not None, "squint-rl distribution metadata is missing"
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts["squint"] == "squint_rl.cli:main"
    main = import_module("squint_rl.cli").main
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "squint 0.1.0"
