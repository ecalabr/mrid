"""The thumbnail pump must apply results on the main thread only.

Worker threads calling root.after() do not merely enqueue: _tkinter forwards the
call to the interpreter's owning thread and blocks the caller until that thread
services Tcl's event queue. A window still being built has not returned to the
event loop, so its own loaders deadlock it — the window opens and no thumbnail
or description ever renders.
"""

import queue
import threading

from mrid_app import gui


class _FakeRoot:
    """Records after() calls and runs them synchronously, noting the thread."""

    def __init__(self):
        self.calls = []
        self.after_threads = []
        self._depth = 0

    def after(self, delay, func=None):
        self.after_threads.append(threading.current_thread().name)
        # Run one generation of the pump so the test makes progress, but do not
        # recurse forever.
        if func is not None and self._depth < 3:
            self._depth += 1
            func()


def test_pump_applies_queued_results():
    root = _FakeRoot()
    results = queue.Queue()
    results.put(("/a.dcm", "IMG-A"))
    results.put(("/b.dcm", "IMG-B"))
    applied = []

    gui._pump_thumbnail_results(root, results, lambda p, i: applied.append((p, i)),
                                threading.Event())
    assert applied == [("/a.dcm", "IMG-A"), ("/b.dcm", "IMG-B")]


def test_pump_stops_when_the_window_closes():
    root = _FakeRoot()
    results = queue.Queue()
    results.put(("/a.dcm", "IMG-A"))
    applied = []
    stop = threading.Event()
    stop.set()

    gui._pump_thumbnail_results(root, results, lambda p, i: applied.append((p, i)), stop)
    assert applied == [], "a closed window must not touch its destroyed widgets"


def test_pump_survives_an_update_that_raises():
    """One bad panel must not stop the remaining thumbnails from rendering."""
    root = _FakeRoot()
    results = queue.Queue()
    results.put(("/bad.dcm", "IMG"))
    results.put(("/good.dcm", "IMG"))
    applied = []

    def _update(path, img):
        if path == "/bad.dcm":
            raise RuntimeError("panel was destroyed")
        applied.append(path)

    gui._pump_thumbnail_results(root, results, _update, threading.Event())
    assert applied == ["/good.dcm"]


def test_pump_reschedules_itself():
    """It must keep polling; results arrive after the first tick."""
    root = _FakeRoot()
    gui._pump_thumbnail_results(root, queue.Queue(), lambda p, i: None,
                                threading.Event())
    assert len(root.after_threads) > 1, "pump did not reschedule"


def test_loaders_never_touch_tk():
    """Guard the invariant directly: no root.after(...) inside a function that
    runs on a pool worker."""
    import inspect

    src = inspect.getsource(gui)
    # The worker-side helper must not reference the Tk root at all.
    start = src.index("def _load_into_queue")
    body = src[start:src.index("_executor = _get_thumbnail_executor()", start)]
    assert "root." not in body, "worker helper touches Tk: " + body


def test_thumbnail_executor_is_created_once_under_concurrency():
    """The lazy global is reached from more than one place; racing creators
    would leak a pool whose threads nothing ever joins."""
    gui._THUMBNAIL_EXECUTOR = None
    seen = []
    barrier = threading.Barrier(8)

    def _grab():
        barrier.wait()
        seen.append(gui._get_thumbnail_executor())

    threads = [threading.Thread(target=_grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(id(e) for e in seen)) == 1, "more than one executor was created"
