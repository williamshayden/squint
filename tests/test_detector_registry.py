from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from edge_perception.detectors.registry import detector_descriptors, load_detector


def test_registry_discovery_does_not_import_model_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))

    descriptors = detector_descriptors()

    assert [item.detector_id for item in descriptors] == ["dfine-nano-coco"]
    assert descriptors[0].display_name == "D-FINE Nano (COCO)"
    assert descriptors[0].model_id == "ustc-community/dfine-nano-coco"
    assert descriptors[0].revision == "066438d3d8f0da137a37b38fdf3368fd4afceced"
    assert imported == []


def test_registry_loads_known_detector_only_after_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, str]] = []

    class FakeDfineDetector:
        @classmethod
        def load(cls, *, threshold: float, device: str) -> object:
            calls.append((threshold, device))
            return object()

    module = ModuleType("edge_perception.detectors.dfine")
    module.DfineDetector = FakeDfineDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_perception.detectors.dfine", module)

    assert load_detector("dfine-nano-coco", threshold=0.3, device="cpu") is not None
    assert calls == [(0.3, "cpu")]


def test_registry_rejects_unknown_detector_without_importing_dfine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "edge_perception.detectors.dfine", raising=False)

    with pytest.raises(ValueError, match="^unknown detector ID: unknown$"):
        load_detector("unknown", threshold=0.3, device="auto")

    assert "edge_perception.detectors.dfine" not in sys.modules
