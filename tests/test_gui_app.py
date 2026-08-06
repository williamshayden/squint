from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
)
from pytestqt.qtbot import QtBot

from edge_perception.contracts import Region
from edge_perception.gui import main_window
from edge_perception.gui.main_window import MainWindow
from edge_perception.gui.region_view import RegionView
from edge_perception.video import DecodedFrame


def _preview_frame(width: int = 200, height: int = 100) -> DecodedFrame:
    return DecodedFrame(
        frame_index=0,
        source_time_ms=0.0,
        image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _stub_first_frame(monkeypatch: pytest.MonkeyPatch, frame: DecodedFrame) -> None:
    monkeypatch.setattr(main_window, "first_video_frame", lambda _path: frame)


def test_main_window_is_one_native_qmain_window(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    assert isinstance(window, QMainWindow)
    assert window.objectName() == "edge-perception-main-window"
    assert window.findChild(QGraphicsView, "source-view") is not None
    assert window.findChild(QPushButton, "run-button") is not None


def test_load_video_replaces_stale_regions_and_shows_source_metadata(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    window = MainWindow()
    qtbot.addWidget(window)
    view = window.findChild(RegionView, "source-view")
    assert view is not None
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    window.load_video(first_path)
    view.add_region(Region("stale", 1, 2, 3, 4))

    window.load_video(second_path)

    path_label = window.findChild(QLabel, "source-path")
    dimensions_label = window.findChild(QLabel, "source-dimensions")
    assert path_label is not None
    assert dimensions_label is not None
    assert path_label.text() == str(second_path)
    assert dimensions_label.text() == "64 × 48 px"
    assert view.scene().sceneRect().width() == 64.0
    assert view.scene().sceneRect().height() == 48.0
    assert view.regions() == ()


def test_numeric_resize_updates_scene_and_contract(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_video(Path("source.mp4"))
    view = window.findChild(RegionView, "source-view")
    region_id = window.findChild(QLineEdit, "region-id")
    x = window.findChild(QSpinBox, "region-x")
    y = window.findChild(QSpinBox, "region-y")
    width = window.findChild(QSpinBox, "region-width")
    height = window.findChild(QSpinBox, "region-height")
    new_region = window.findChild(QPushButton, "new-region-button")
    assert all(widget is not None for widget in (view, region_id, x, y, width, height, new_region))
    assert view is not None
    assert region_id is not None
    assert x is not None
    assert y is not None
    assert width is not None
    assert height is not None
    assert new_region is not None
    region_id.setText("roi")
    x.setValue(10)
    y.setValue(20)
    width.setValue(40)
    height.setValue(30)
    new_region.click()

    x.setValue(25)
    y.setValue(15)
    width.setValue(80)
    with qtbot.waitSignal(view.regionsChanged) as changed:
        height.setValue(45)

    assert (x.value(), y.value(), width.value(), height.value()) == (25, 15, 80, 45)
    assert view.regions() == (Region("roi", 25, 15, 80, 45),)
    assert changed.args == [(Region("roi", 25, 15, 80, 45),)]
    item = next(
        item
        for item in view.scene().items()
        if isinstance(item, QGraphicsRectItem) and item.data(0) == "roi"
    )
    assert item.mapRectToScene(item.rect()) == item.rect().translated(25.0, 15.0)


def test_region_controls_add_and_delete_selected_region(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_video(Path("source.mp4"))
    region_id = window.findChild(QLineEdit, "region-id")
    new_region = window.findChild(QPushButton, "new-region-button")
    delete_region = window.findChild(QPushButton, "delete-region-button")
    view = window.findChild(RegionView, "source-view")
    assert region_id is not None
    assert new_region is not None
    assert delete_region is not None
    assert view is not None
    region_id.setText("roi")

    new_region.click()
    delete_region.click()

    assert view.regions() == ()
    assert all(item.data(0) != "roi" for item in view.scene().items())


def test_run_enables_only_with_source_and_output(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    run = window.findChild(QPushButton, "run-button")
    output = window.findChild(QLineEdit, "output")
    assert run is not None
    assert output is not None

    output.setText(str(tmp_path / "runs"))
    assert run.isEnabled() is False
    window.load_video(tmp_path / "source.mp4")
    assert run.isEnabled() is True
    output.clear()
    assert run.isEnabled() is False


def test_file_menu_opens_selected_video(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    selected_path = tmp_path / "selected.mp4"
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(selected_path), "Video files (*.mp4)"),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    action = window.findChild(QAction, "open-video-action")
    assert action is not None
    assert action.text() == "Open Video…"

    action.trigger()

    path_label = window.findChild(QLabel, "source-path")
    assert path_label is not None
    assert path_label.text() == str(selected_path)
