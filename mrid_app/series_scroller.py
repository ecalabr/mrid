"""
In-place mouse-wheel scrolling through a DICOM series on a thumbnail widget.

A SeriesScroller attaches to an existing thumbnail Label. Until the user
scrolls, it does nothing — the cached center-slice thumbnail stays in place
and initial window load time is unaffected. On the first scroll event the
series files are enumerated and sorted in a background thread; subsequent
scrolls step through the slices, which are loaded from the DICOM source on
demand and kept in a small in-memory LRU cache.
"""

import threading
from collections import OrderedDict

from PIL import Image, ImageDraw, ImageTk

from .logging_config import get_logger

logger = get_logger()

# Max decoded slices kept in memory per series
_CACHE_SIZE = 128


class SeriesScroller:
    def __init__(self, panel, dicom_path, display_size):
        """
        Args:
            panel: tk.Label displaying the series thumbnail
            dicom_path: path to any DICOM file of the series
            display_size: pixel size of the displayed (square) thumbnail
        """
        self.panel = panel
        self.dicom_path = dicom_path
        self.display_size = display_size

        self._files = None          # sorted series file paths (set by worker)
        self._index = None          # currently requested slice index
        self._pending_steps = 0     # scrolls received before enumeration finished
        self._cache = OrderedDict() # index -> PIL.Image
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._worker = None
        self._alive = True

        panel.bind("<MouseWheel>", self._on_mousewheel, add="+")
        panel.bind("<Button-4>", lambda e: self._step(-1), add="+")  # Linux up
        panel.bind("<Button-5>", lambda e: self._step(1), add="+")   # Linux down
        panel.bind("<Destroy>", self._on_destroy, add="+")

    # ── Event handling ────────────────────────────────────────────────────────

    def _on_mousewheel(self, event):
        # Windows/macOS deliver signed deltas; scroll down -> next slice
        self._step(1 if event.delta < 0 else -1)
        return "break"

    def _on_destroy(self, _event=None):
        self._alive = False
        self._wake.set()

    def _step(self, direction):
        with self._lock:
            if self._worker is None:
                # First scroll: spin up the background loader
                self._worker = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker.start()
            if self._files is not None:
                self._index = max(0, min(self._index + direction, len(self._files) - 1))
            else:
                # Series not enumerated yet — remember the scrolls for later
                self._pending_steps += direction
        self._wake.set()
        return "break"

    # ── Background loading ────────────────────────────────────────────────────

    def _worker_loop(self):
        from . import processing

        try:
            files = processing.get_sorted_series_files([self.dicom_path])
        except Exception as e:
            logger.debug(f"SeriesScroller: could not enumerate series for {self.dicom_path}: {e}")
            return
        if not files:
            return

        with self._lock:
            self._files = files
            center = len(files) // 2
            self._index = max(0, min(center + self._pending_steps, len(files) - 1))
            self._pending_steps = 0

        while self._alive:
            with self._lock:
                index = self._index
            img = self._get_slice(index)
            if img is not None:
                self._show(img, index)
            # Wait until the index changes (or widget is destroyed)
            self._wake.clear()
            with self._lock:
                if self._index == index:
                    wait_needed = True
                else:
                    wait_needed = False
            if wait_needed:
                self._wake.wait()

    def _get_slice(self, index):
        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
            path = self._files[index]
        from .processing import _load_single_slice
        img = _load_single_slice(path)
        if img is None:
            return None
        img = img.convert("RGB").resize((self.display_size, self.display_size))
        with self._lock:
            self._cache[index] = img
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return img

    # ── Display ───────────────────────────────────────────────────────────────

    def _annotate(self, img, index):
        """Stamp the slice counter into the bottom-left corner."""
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        text = f"{index + 1}/{len(self._files)}"
        pad = max(2, self.display_size // 64)
        x, y = pad, self.display_size - pad
        try:
            bbox = draw.textbbox((x, y), text, anchor="lb")
            draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                           fill=(0, 0, 0))
            draw.text((x, y), text, fill=(255, 255, 0), anchor="lb")
        except Exception:
            draw.text((x, y - 10), text, fill=(255, 255, 0))
        return annotated

    def _show(self, img, index):
        def update():
            try:
                if not self.panel.winfo_exists():
                    return
                # Drop stale frames if the user kept scrolling
                with self._lock:
                    if self._index != index:
                        return
                tk_img = ImageTk.PhotoImage(self._annotate(img, index), master=self.panel)
                self.panel.configure(image=tk_img)
                self.panel.image = tk_img
            except Exception as e:
                logger.debug(f"SeriesScroller display update failed: {e}")

        try:
            self.panel.after(0, update)
        except Exception:
            pass


def attach_series_scrolling(panel, dicom_path, display_size):
    """Attach mouse-wheel series scrolling to a thumbnail Label.

    Returns the SeriesScroller (kept alive by the panel's attribute)."""
    if not dicom_path:
        return None
    try:
        scroller = SeriesScroller(panel, dicom_path, display_size)
        panel._series_scroller = scroller
        return scroller
    except Exception as e:
        logger.debug(f"Could not attach series scrolling: {e}")
        return None
