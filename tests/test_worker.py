from threading import Event, get_ident

from starlab.bootstrap import create_window


def test_timed_out_close_finishes_automatically_and_cleans_up_on_job_thread(qtbot, monkeypatch):
    window = create_window()
    qtbot.addWidget(window)
    window.show()
    runner = window.controller.runner
    release, started = Event(), Event()
    threads, cleanup_threads = [], []

    def operation(cancel):
        threads.append(get_ident())
        started.set()
        release.wait(2)

    runner.cleanup = lambda: cleanup_threads.append(get_ident())
    runner.submit("preview", operation)
    assert started.wait(1)
    wait = runner.pool.waitForDone
    monkeypatch.setattr(runner.pool, "waitForDone", lambda _: wait(0))
    try:
        window.close()
        assert window.isVisible()
        window.close()  # A second close must not schedule cleanup twice.
        release.set()
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=1500)
        assert cleanup_threads == threads
        assert runner.closing
    finally:
        release.set()
        assert wait(2000)
