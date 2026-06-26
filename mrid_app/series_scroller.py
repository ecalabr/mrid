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
        self._user_scrolled = False  # only annotate slice counter after a scroll
        self._window = None          # applied window: None (DICOM default) or (wc, ww)
        self._window_request = None  # pending 'auto' / 'default' window change

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
            self._user_scrolled = True
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

    def set_window(self, mode):
        """Request an 'auto' or 'default' window/level. The change is computed in
        the worker thread and re-renders the current slice; it then applies to all
        slices while scrolling. Safe to call before the user has scrolled."""
        with self._lock:
            self._window_request = mode
            if self._worker is None:
                self._worker = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker.start()
        self._wake.set()

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
            # Apply any pending window/level change before rendering.
            with self._lock:
                req = self._window_request
                self._window_request = None
            if req is not None:
                self._apply_window_request(req)

            with self._lock:
                index = self._index
            img = self._get_slice(index)
            if img is not None:
                self._show(img, index)
            # Wait until the index changes, a window change is requested, or the
            # widget is destroyed.
            self._wake.clear()
            with self._lock:
                wait_needed = (self._index == index and self._window_request is None)
            if wait_needed:
                self._wake.wait()

    def _apply_window_request(self, req):
        from . import processing
        new_window = None
        if req == 'auto':
            with self._lock:
                idx = self._index if self._index is not None else (len(self._files) // 2)
                path = self._files[idx] if self._files else self.dicom_path
            arr = processing.read_slice_array(path)
            new_window = processing.compute_auto_window(arr)
        # 'default' -> new_window stays None (DICOM default)
        with self._lock:
            self._window = new_window
            self._cache.clear()

    def _get_slice(self, index):
        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
            path = self._files[index]
            window = self._window
        from .processing import _load_single_slice
        img = _load_single_slice(path, window=window)
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
                # Only stamp the slice counter once the user has actually
                # scrolled (a window/level change alone shouldn't add it).
                shown = self._annotate(img, index) if self._user_scrolled else img
                tk_img = ImageTk.PhotoImage(shown, master=self.panel)
                self.panel.configure(image=tk_img)
                self.panel.image = tk_img
            except Exception as e:
                logger.debug(f"SeriesScroller display update failed: {e}")

        try:
            self.panel.after(0, update)
        except Exception:
            pass


def _attach_window_menu(panel, scroller):
    """Right-click menu on a thumbnail to auto- or default-window/level it."""
    import sys
    import tkinter as tk
    try:
        from . import ui_theme
    except Exception:
        ui_theme = None

    menu = tk.Menu(panel, tearoff=0)
    if ui_theme is not None:
        try:
            ui_theme.style_menu(menu)
        except Exception:
            pass
    menu.add_command(label="Auto window/level", command=lambda: scroller.set_window('auto'))
    menu.add_command(label="Default window/level", command=lambda: scroller.set_window('default'))

    def _popup(event):
        # Post the menu on button *release* (not press) and let the menu keep its
        # own grab: opening on press and releasing the grab immediately caused the
        # pending button-release to dismiss the menu before a choice could be
        # made. Posting on release leaves the menu open so a second click (or a
        # hold-and-release onto an item) confirms the choice.
        try:
            menu.tk_popup(event.x_root, event.y_root)
        except Exception:
            try:
                menu.grab_release()
            except Exception:
                pass
        return "break"

    panel.bind("<ButtonRelease-3>", _popup, add="+")
    if sys.platform == "darwin":
        # macOS commonly delivers right-click as Button-2 / Control-click.
        panel.bind("<ButtonRelease-2>", _popup, add="+")
        panel.bind("<Control-ButtonRelease-1>", _popup, add="+")


def attach_series_scrolling(panel, dicom_path, display_size):
    """Attach mouse-wheel series scrolling to a thumbnail Label.

    Returns the SeriesScroller (kept alive by the panel's attribute)."""
    if not dicom_path:
        return None
    try:
        scroller = SeriesScroller(panel, dicom_path, display_size)
        panel._series_scroller = scroller
        _attach_window_menu(panel, scroller)
        return scroller
    except Exception as e:
        logger.debug(f"Could not attach series scrolling: {e}")
        return None
