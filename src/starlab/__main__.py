"""Parse the optional input path and run the standalone Qt application."""

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .bootstrap import create_window


def main() -> int:
    """Create the GUI, optionally open an image, and return the Qt event-loop exit code."""
    parser = argparse.ArgumentParser(
        description="Star Effects Lab — standalone TIFF diffraction effects"
    )
    parser.add_argument("image", nargs="?", help="Optional TIFF image to open")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Star Effects Lab")
    app.setOrganizationName("Star Effects Lab")
    window = create_window()
    window.show()
    if args.image and window.controller is not None:
        window.controller.open_path(args.image)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
