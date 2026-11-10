from collections.abc import Callable
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from ...domain.imaging import Cancelled


class JobSignals(QObject):
    finished = Signal(int, str, object, object)


class Job(QRunnable):
    def __init__(self, number: int, kind: str, operation: Callable):
        super().__init__()
        self.number, self.kind, self.operation = number, kind, operation
        self.cancel = Event()
        self.signals = JobSignals()

    def run(self) -> None:
        result, error = None, None
        try:
            result = self.operation(self.cancel)
        except Exception as exc:
            error = exc
        self.signals.finished.emit(self.number, self.kind, result, error)


class JobRunner(QObject):
    """One active worker and one replaceable pending request. Only current results escape."""

    completed = Signal(str, object)
    failed = Signal(str)
    activity = Signal(str)
    idle = Signal()
    shutdown_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.pool.setExpiryTimeout(-1)  # GPU context remains on its creating job thread.
        self.cleanup = None
        self.active: Job | None = None
        self.pending: Job | None = None
        self.serial = 0
        self.closing = False
        self.shutdown_timer = QTimer(self)
        self.shutdown_timer.setInterval(25)
        self.shutdown_timer.timeout.connect(self._check_shutdown)

    @property
    def busy(self) -> bool:
        return self.active is not None or self.pending is not None

    def submit(self, kind: str, operation: Callable) -> None:
        if self.closing:
            return
        self.serial += 1
        job = Job(self.serial, kind, operation)
        if self.active is not None:
            self.active.cancel.set()
            self.pending = job
        else:
            self._start(job)

    def _start(self, job: Job) -> None:
        self.active = job
        job.signals.finished.connect(self._finished)
        self.activity.emit(job.kind)
        self.pool.start(job)

    @Slot(int, str, object, object)
    def _finished(self, number: int, kind: str, result, error) -> None:
        self.active = None
        if number == self.serial and not self.closing:
            if error is None:
                self.completed.emit(kind, result)
            elif not isinstance(error, Cancelled):
                self.failed.emit(str(error) or type(error).__name__)
        # completed slots may have started another job.
        if self.active is None and self.pending is not None and not self.closing:
            pending, self.pending = self.pending, None
            self._start(pending)
        if not self.busy and not self.closing:
            self.idle.emit()

    def cancel(self) -> None:
        self.serial += 1
        self.pending = None
        if self.active:
            self.active.cancel.set()

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        """Cancel once, queue thread-owned cleanup once, and notify after a timed-out drain."""
        if not self.closing:
            self.closing = True
            self.cancel()
            if self.cleanup is not None:
                cleanup, self.cleanup = self.cleanup, None
                self.pool.start(Job(0, "cleanup", lambda cancel: cleanup()))
        finished = self.pool.waitForDone(timeout_ms)
        if finished:
            self.shutdown_timer.stop()
        else:
            self.shutdown_timer.start()
        return finished

    def _check_shutdown(self) -> None:
        """Finish a deferred close only after jobs and GPU cleanup leave the pool."""
        if self.pool.waitForDone(0):
            self.shutdown_timer.stop()
            self.shutdown_finished.emit()
