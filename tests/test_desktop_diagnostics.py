"""Session diagnostics retain failures without making unavailable log storage fatal."""

import faulthandler
import logging

import pytest

from starlab.infrastructure.diagnostics import application_diagnostics


def test_uncaught_failure_is_recorded_and_original_exception_survives(tmp_path):
    was_enabled = faulthandler.is_enabled()
    original_handlers = tuple(logging.getLogger("starlab").handlers)
    with pytest.raises(RuntimeError, match="startup failure"):
        with application_diagnostics(tmp_path) as logger:
            logger.info("Loading the first image")
            raise RuntimeError("startup failure")
    contents = (tmp_path / "startup.log").read_text(encoding="utf-8")
    assert "Loading the first image" in contents
    assert "RuntimeError: startup failure" in contents
    assert tuple(logging.getLogger("starlab").handlers) == original_handlers
    assert faulthandler.is_enabled() == was_enabled


def test_unavailable_log_directory_does_not_prevent_running_application(tmp_path):
    blocked = tmp_path / "file-instead-of-directory"
    blocked.write_text("existing data", encoding="utf-8")
    completed = False
    with application_diagnostics(blocked):
        completed = True
    assert completed
    assert blocked.read_text(encoding="utf-8") == "existing data"
