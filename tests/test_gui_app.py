from __future__ import annotations

from PySide6.QtWidgets import QGraphicsView, QMainWindow, QPushButton
from pytestqt.qtbot import QtBot

from edge_perception.gui.main_window import MainWindow


def test_main_window_is_one_native_qmain_window(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    assert isinstance(window, QMainWindow)
    assert window.objectName() == "edge-perception-main-window"
    assert window.findChild(QGraphicsView, "source-view") is not None
    assert window.findChild(QPushButton, "run-button") is not None
