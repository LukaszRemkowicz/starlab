"""Cooperative latest-only jobs with GUI-thread delivery and thread-owned cleanup."""

from collections.abc import Callable
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from ...domain.imaging import Cancelled


class JobSignals(QObject):
    """Deliver an operation's opaque result or exception to the runner's GUI thread."""

    finished = Signal(int, str, object, object)


class Job(QRunnable):
    """Run one immutable operation snapshot with a cooperative cancellation event."""

    def __init__(self, number: int, kind: str, operation: Callable[[Event], object]) -> None:
        """Retain the operation and create signals on the submitting GUI thread."""
        super().__init__()
        self.number, self.kind, self.operation = number, kind, operation
        self.cancel = Event()
        self.signals = JobSignals()

    def run(self) -> None:
        """Execute on the pool thread and always report completion to the runner."""
        result: object = None
        error: Exception | None = None
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

    def __init__(self, parent: QObject | None = None) -> None:
        """Own one persistent worker thread so GPU resources never migrate threads."""
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.pool.setExpiryTimeout(-1)  # GPU context remains on its creating job thread.
        self.cleanup: Callable[[], None] | None = None
        self.active: Job | None = None
        self.pending: Job | None = None
        self.serial = 0
        self.closing = False
        self.shutdown_timer = QTimer(self)
        self.shutdown_timer.setInterval(25)
        self.shutdown_timer.timeout.connect(self._check_shutdown)

    @property
    def busy(self) -> bool:
        """Whether a worker or replaceable pending operation remains outstanding."""
        return self.active is not None or self.pending is not None

    def submit(self, kind: str, operation: Callable[[Event], object]) -> None:
        """Replace pending work and cancel older results without blocking the GUI."""
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
        """Connect GUI-owned completion delivery before scheduling the runnable."""
        self.active = job
        job.signals.finished.connect(self._finished)
        self.activity.emit(job.kind)
        self.pool.start(job)

    @Slot(int, str, object, object)
    def _finished(self, number: int, kind: str, result: object, error: Exception | None) -> None:
        """Suppress obsolete results, start pending work and handle reentrant submissions."""
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
        """Invalidate active results and discard pending work immediately."""
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
