from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem
from pytestqt.qtbot import QtBot

from edge_perception.contracts import Region
from edge_perception.gui.region_view import RegionView


def _region_item(view: RegionView, region_id: str) -> QGraphicsRectItem:
    return next(
        item
        for item in view.scene().items()
        if isinstance(item, QGraphicsRectItem) and item.data(0) == region_id
    )


def _source_rect(item: QGraphicsRectItem) -> QRectF:
    return item.mapRectToScene(item.rect())


def test_region_view_keeps_scene_coordinates_in_source_pixels(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    view.set_rgb_frame(image)

    view.add_region(Region("upper-right", 960, 0, 960, 540))

    assert view.scene().sceneRect() == QRectF(0.0, 0.0, 1920.0, 1080.0)
    assert view.regions() == (Region("upper-right", 960, 0, 960, 540),)


def test_region_from_drag_clips_and_rounds_to_source_pixels(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    view.set_rgb_frame(np.zeros((100, 200, 3), dtype=np.uint8))

    region = view.region_from_scene_rect("roi", QRectF(-2.4, 10.2, 44.8, 30.1))

    assert region == Region("roi", 0, 10, 43, 31)


def test_region_ids_must_be_unique_and_not_reserved(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    view.set_rgb_frame(np.zeros((100, 200, 3), dtype=np.uint8))
    view.add_region(Region("roi", 10, 10, 20, 20))
    item_count = len(view.scene().items())

    with pytest.raises(ValueError, match="unique"):
        view.add_region(Region("roi", 40, 40, 20, 20))
    with pytest.raises(ValueError, match="reserved"):
        view.add_region(Region("full-frame", 0, 0, 200, 100))

    assert view.regions() == (Region("roi", 10, 10, 20, 20),)
    assert len(view.scene().items()) == item_count


def test_zero_size_drag_creates_no_region(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    view.set_rgb_frame(np.zeros((100, 200, 3), dtype=np.uint8))
    item_count = len(view.scene().items())

    region = view.region_from_scene_rect("roi", QRectF(20.0, 30.0, 0.0, 0.0))

    assert region is None
    assert view.regions() == ()
    assert len(view.scene().items()) == item_count


def test_moving_region_clamps_to_source_bounds(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    view.set_rgb_frame(np.zeros((100, 200, 3), dtype=np.uint8))
    view.add_region(Region("roi", 10, 20, 40, 20))
    item = _region_item(view, "roi")

    with qtbot.waitSignal(view.regionsChanged) as changed:
        item.setPos(QPointF(190.0, 95.0))

    assert _source_rect(item) == QRectF(160.0, 80.0, 40.0, 20.0)
    assert view.regions() == (Region("roi", 160, 80, 40, 20),)
    assert changed.args == [(Region("roi", 160, 80, 40, 20),)]


def test_delete_removes_selected_region(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    view.set_rgb_frame(np.zeros((100, 200, 3), dtype=np.uint8))
    view.add_region(Region("first", 10, 10, 20, 20))
    view.add_region(Region("second", 40, 40, 20, 20))
    _region_item(view, "first").setSelected(True)

    with qtbot.waitSignal(view.regionsChanged) as changed:
        view.delete_selected_region()

    assert view.regions() == (Region("second", 40, 40, 20, 20),)
    assert all(item.data(0) != "first" for item in view.scene().items())
    assert changed.args == [(Region("second", 40, 40, 20, 20),)]


def test_resize_only_refits_view(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    view.resize(400, 300)
    view.set_rgb_frame(np.zeros((100, 200, 3), dtype=np.uint8))
    view.add_region(Region("roi", 10, 20, 40, 20))
    before = view.regions()

    view.resize(800, 250)

    assert view.regions() == before
    assert _source_rect(_region_item(view, "roi")) == QRectF(10.0, 20.0, 40.0, 20.0)


def test_rgb_frame_is_owned_by_qimage(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    view.set_rgb_frame(image)

    image[0, 0] = (200, 210, 220)

    pixmap_item = next(
        item for item in view.scene().items() if isinstance(item, QGraphicsPixmapItem)
    )
    pixel = pixmap_item.pixmap().toImage().pixelColor(0, 0)
    assert (pixel.red(), pixel.green(), pixel.blue()) == (10, 20, 30)
