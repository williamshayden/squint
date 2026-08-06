"""Source-pixel preview and region overlays for the optional Qt GUI."""

from __future__ import annotations

from math import ceil, floor
from typing import Any, override

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from edge_perception.contracts import Region


class _RegionItem(QGraphicsRectItem):
    """Movable overlay whose position remains on bounded source pixels."""

    def __init__(self, owner: RegionView, region: Region) -> None:
        super().__init__(0.0, 0.0, float(region.width), float(region.height))
        self._owner = owner
        self._notify_geometry_changes = False
        self.setData(0, region.region_id)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(float(region.x), float(region.y))
        self._notify_geometry_changes = True

    def set_source_region(self, region: Region) -> None:
        """Replace logical geometry without exposing intermediate states."""

        self._notify_geometry_changes = False
        self.setRect(0.0, 0.0, float(region.width), float(region.height))
        self.setPos(float(region.x), float(region.y))
        self._notify_geometry_changes = True

    @override
    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene() is not None:
            proposed = value
            if isinstance(proposed, QPointF):
                bounds = self.scene().sceneRect()
                maximum_x = bounds.right() - self.rect().width()
                maximum_y = bounds.bottom() - self.rect().height()
                return QPointF(
                    min(max(float(floor(proposed.x())), bounds.left()), maximum_x),
                    min(max(float(floor(proposed.y())), bounds.top()), maximum_y),
                )
        changed = super().itemChange(change, value)
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and self._notify_geometry_changes
        ):
            self._owner._region_item_changed()
        return changed


class RegionView(QGraphicsView):
    """Display one RGB frame and editable overlays in source-pixel coordinates."""

    regionsChanged = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("source-view")
        self._source_scene = QGraphicsScene(self)
        self.setScene(self._source_scene)
        self._image: QImage | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._region_items: list[_RegionItem] = []

    def set_rgb_frame(self, image: np.ndarray) -> None:
        """Display an RGB array after copying its pixel storage into Qt."""

        height, width, channels = image.shape
        if image.dtype != np.uint8 or channels != 3:
            raise ValueError("RGB frame must have shape (height, width, 3) and dtype uint8")

        rgb = np.ascontiguousarray(image)
        self._image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._source_scene.clear()
        self._region_items.clear()
        self._pixmap_item = self._source_scene.addPixmap(QPixmap.fromImage(self._image))
        self._source_scene.setSceneRect(QRectF(0.0, 0.0, float(width), float(height)))
        self._fit_source()

    def add_region(self, region: Region) -> None:
        """Add a named source-pixel rectangle in insertion order."""

        region_ids = {existing.region_id for existing in self.regions()}
        if region.region_id == "full-frame":
            raise ValueError("full-frame is a reserved region ID")
        if region.region_id in region_ids:
            raise ValueError("region IDs must be unique")
        bounds = self._source_scene.sceneRect()
        if (
            bounds.isEmpty()
            or region.x < 0
            or region.y < 0
            or region.x + region.width > int(bounds.width())
            or region.y + region.height > int(bounds.height())
        ):
            raise ValueError("region must be fully inside the source frame")

        item = _RegionItem(self, region)
        self._source_scene.addItem(item)
        self._region_items.append(item)
        self.regionsChanged.emit(self.regions())

    def regions(self) -> tuple[Region, ...]:
        """Return immutable source-pixel regions in insertion order."""

        return tuple(self._region_from_item(item) for item in self._region_items)

    def delete_selected_region(self) -> None:
        """Delete selected overlays and emit the remaining region tuple."""

        selected = [item for item in self._region_items if item.isSelected()]
        if not selected:
            return
        for item in selected:
            self._source_scene.removeItem(item)
            self._region_items.remove(item)
        self.regionsChanged.emit(self.regions())

    def selected_region(self) -> Region | None:
        """Return the selected region, if any."""

        return next(
            (self._region_from_item(item) for item in self._region_items if item.isSelected()),
            None,
        )

    def select_region(self, region_id: str) -> None:
        """Select one overlay by ID."""

        self._source_scene.clearSelection()
        item = self._item_for_id(region_id)
        item.setSelected(True)

    def update_region(
        self,
        region_id: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Region:
        """Apply exact numeric geometry, clamped to the source frame."""

        item = self._item_for_id(region_id)
        bounds = self._source_scene.sceneRect()
        source_width = int(bounds.width())
        source_height = int(bounds.height())
        clamped_width = min(max(width, 1), source_width)
        clamped_height = min(max(height, 1), source_height)
        clamped_x = min(max(x, 0), source_width - clamped_width)
        clamped_y = min(max(y, 0), source_height - clamped_height)
        updated = Region(
            region_id,
            clamped_x,
            clamped_y,
            clamped_width,
            clamped_height,
        )
        item.set_source_region(updated)
        self.regionsChanged.emit(self.regions())
        return updated

    def region_from_scene_rect(self, region_id: str, rect: QRectF) -> Region | None:
        """Normalize, clip, and outward-round a scene rectangle to source pixels."""

        bounds = self._source_scene.sceneRect()
        normalized = rect.normalized()
        left = max(bounds.left(), normalized.left())
        top = max(bounds.top(), normalized.top())
        right = min(bounds.right(), normalized.right())
        bottom = min(bounds.bottom(), normalized.bottom())
        pixel_left = floor(left)
        pixel_top = floor(top)
        pixel_right = ceil(right)
        pixel_bottom = ceil(bottom)
        if pixel_right <= pixel_left or pixel_bottom <= pixel_top:
            return None
        return Region(
            region_id,
            pixel_left,
            pixel_top,
            pixel_right - pixel_left,
            pixel_bottom - pixel_top,
        )

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_source()

    def _fit_source(self) -> None:
        scene_rect = self._source_scene.sceneRect()
        if not scene_rect.isEmpty():
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _region_from_item(self, item: _RegionItem) -> Region:
        return Region(
            str(item.data(0)),
            int(item.pos().x()),
            int(item.pos().y()),
            int(item.rect().width()),
            int(item.rect().height()),
        )

    def _region_item_changed(self) -> None:
        self.regionsChanged.emit(self.regions())

    def _item_for_id(self, region_id: str) -> _RegionItem:
        try:
            return next(item for item in self._region_items if item.data(0) == region_id)
        except StopIteration as error:
            raise KeyError(region_id) from error
