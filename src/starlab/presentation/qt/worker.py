from collections.abc import Callable
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

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
        if not self.busy:
            self.idle.emit()

    def cancel(self) -> None:
        self.serial += 1
        self.pending = None
        if self.active:
            self.active.cancel.set()

    def shutdown(self) -> bool:
        self.closing = True
        self.cancel()
        if self.cleanup is not None:
            cleanup, self.cleanup = self.cleanup, None
            self.pool.start(Job(0, "cleanup", lambda cancel: cleanup()))
        return self.pool.waitForDone(5000)
