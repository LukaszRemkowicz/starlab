import argparse
import sys

from PySide6.QtWidgets import QApplication

from .bootstrap import create_window


def main() -> int:
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
    if args.image:
        window.controller.open_path(args.image)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
