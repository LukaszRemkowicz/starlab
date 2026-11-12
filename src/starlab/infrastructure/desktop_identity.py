"""Native desktop process identity without importing GUI components."""

import ctypes
import logging
import sys

logger = logging.getLogger(__name__)


def set_process_identity(application_id: str) -> None:
    """Give Windows taskbar grouping a stable application identity before window creation."""
    # A positive platform block lets mypy skip Windows-only ctypes names on other systems.
    if sys.platform == "win32":
        shell: ctypes.WinDLL = ctypes.WinDLL("shell32", use_last_error=True)
        shell.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        shell.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        result: int = shell.SetCurrentProcessExplicitAppUserModelID(application_id)
        if result < 0:
            logger.warning("Windows application identity could not be set: HRESULT %#x", result)
