import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-gpu", action="store_true", help="Run native OpenGL integration tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires --run-gpu and a native OpenGL 3.3 context")
    config.addinivalue_line(
        "markers", "slow: numerical correctness checks allowing cold JIT compilation"
    )
    if config.getoption("--run-gpu") and os.environ["QT_QPA_PLATFORM"] == "offscreen":
        os.environ["QT_QPA_PLATFORM"] = {"win32": "windows", "darwin": "cocoa"}.get(
            sys.platform, "xcb"
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-gpu"):
        skip = pytest.mark.skip(reason="Native OpenGL integration requires --run-gpu")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
