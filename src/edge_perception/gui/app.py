"""Native Qt application lifecycle for the optional GUI extra."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtWidgets import QApplication

from edge_perception.gui.main_window import MainWindow

_OPEN_WINDOWS: list[MainWindow] = []


def _forget_window(window: MainWindow) -> None:
    try:
        _OPEN_WINDOWS.remove(window)
    except ValueError:
        pass


def launch_gui(run_dir: Path | None = None, *, argv: Sequence[str] = ()) -> int:
    """Show the native window and run the event loop when this function owns it."""

    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication(["edge-perception", *argv])
        app.setApplicationName("Edge Perception")
    window = MainWindow(run_dir=run_dir)
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda _destroyed=None: _forget_window(window))
    window.show()
    if not owns_application:
        return 0
    return int(app.exec())
