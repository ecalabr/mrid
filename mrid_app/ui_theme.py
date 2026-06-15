"""
Centralized CustomTkinter theming, scaling, and window placement for MRID.

All MRID windows share:
- A dark appearance that renders identically on Windows, macOS, and Linux.
- Screen-proportional scaling: the UI is designed against a 1920x1080
  reference and scaled so it occupies the same fraction of any screen.
- Deterministic cascaded placement of child windows relative to their parent.
"""

import customtkinter as ctk

from . import config
from .logging_config import get_logger

logger = get_logger()

# Reference screen size the UI was designed against
BASE_SCREEN_W = 1920
BASE_SCREEN_H = 1080

# Pixel offset between a child window and its parent in the cascade
CASCADE_STEP = 48

# ── Dark palette accents ──────────────────────────────────────────────────────
# (base, hover) pairs for state-colored buttons
GREEN = "#2e8b46"
GREEN_HOVER = "#27753b"
RED = "#b03a3a"
RED_HOVER = "#963131"
YELLOW = "#9c8a1c"
YELLOW_HOVER = "#857517"
NEUTRAL = "#4a4a4a"
NEUTRAL_HOVER = "#5a5a5a"
DISABLED = "#333333"

# Background for image thumbnail panels (plain tk Labels)
PANEL_BG = "#1a1a1a"
# Tooltip colors
TOOLTIP_BG = "#2b2b2b"
TOOLTIP_FG = "#e0e0e0"

# Dark colors for classic tk.Menu widgets (the menubar and popup menus). On
# Linux these render via Tk and would otherwise be light; on macOS the native
# menubar ignores them.
MENU_BG = "#2b2b2b"
MENU_FG = "#dce4ee"
MENU_ACTIVE_BG = "#1f6aa5"
MENU_ACTIVE_FG = "#ffffff"
MENU_DISABLED_FG = "#777777"


def style_menu(menu):
    """Apply the dark theme to a classic tk.Menu (menubar or popup)."""
    try:
        menu.configure(
            bg=MENU_BG, fg=MENU_FG,
            activebackground=MENU_ACTIVE_BG, activeforeground=MENU_ACTIVE_FG,
            disabledforeground=MENU_DISABLED_FG,
            activeborderwidth=0, borderwidth=0, relief="flat",
        )
    except Exception:
        pass

_SCALE_FACTOR = None


def _ensure_linux_shape_font():
    """Make CustomTkinter's anti-aliased corners work on Linux.

    Smooth widget corners on Tk/X11 are only possible via text (Xft) rendering,
    which CustomTkinter does with a bundled shapes font. If that font isn't
    registered with fontconfig, CTk silently falls back to 'circle_shapes',
    which draws blocky corners. Register the font with fontconfig and force the
    font_shapes method. Best-effort and Linux-only (no effect elsewhere)."""
    import sys
    if not sys.platform.startswith("linux"):
        return
    try:
        import os
        import shutil
        import subprocess

        ctk_dir = os.path.dirname(ctk.__file__)
        shapes_font = os.path.join(ctk_dir, "assets", "fonts",
                                   "CustomTkinter_shapes_font.otf")
        if not os.path.isfile(shapes_font):
            return

        target_dir = os.path.expanduser("~/.fonts")
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, "CustomTkinter_shapes_font.otf")
        if not os.path.isfile(dest):
            shutil.copy(shapes_font, dest)

        # Refresh the fontconfig cache so the running Tk/Xft can find the font.
        try:
            subprocess.run(["fc-cache", "-f", target_dir],
                           check=False, capture_output=True, timeout=30)
        except Exception:
            pass

        # Only force font_shapes if fontconfig can now see the font; otherwise
        # forcing it would render corners as missing glyphs (worse).
        registered = False
        try:
            out = subprocess.run(["fc-list"], check=False, capture_output=True,
                                 timeout=30, text=True)
            registered = "CustomTkinter_shapes_font" in (out.stdout or "")
        except Exception:
            registered = False

        if registered:
            from customtkinter.windows.widgets.core_rendering import DrawEngine
            DrawEngine.preferred_drawing_method = "font_shapes"
            logger.debug("Linux: using font_shapes for smooth widget corners")
    except Exception as exc:
        logger.debug(f"Could not set up Linux shape font: {exc}")


def init_appearance():
    """Apply the global dark theme. Call once before creating any window."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    _ensure_linux_shape_font()


def init_scaling(window):
    """
    Compute the screen-proportional scale factor from the screen size and
    apply it to all CustomTkinter widgets, fonts, and window geometry.
    Call once on the root window.
    """
    global _SCALE_FACTOR
    if _SCALE_FACTOR is not None:
        return _SCALE_FACTOR
    try:
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        scale = min(screen_w / BASE_SCREEN_W, screen_h / BASE_SCREEN_H)
        scale = max(0.6, min(scale, 3.0))
    except Exception:
        scale = 1.0
    _SCALE_FACTOR = scale
    try:
        ctk.set_widget_scaling(scale)
        ctk.set_window_scaling(scale)
    except Exception:
        logger.debug("Could not apply CustomTkinter scaling")
    return scale


def get_scale_factor():
    return _SCALE_FACTOR if _SCALE_FACTOR is not None else 1.0


def scale_value(value):
    """Scale a raw pixel value (pad, offset, image size) by the screen factor.

    Use only for values CustomTkinter does not scale itself (pack/grid padding,
    PIL image sizes, window positions). Widget sizes and fonts passed to CTk
    widgets are scaled automatically and must NOT go through this."""
    return int(round(value * get_scale_factor()))


def scale_thumbnail(value):
    """Scale a thumbnail pixel size by the screen factor and the
    user-configurable THUMBNAIL_SCALE_FACTOR (read live from config)."""
    try:
        tsf = config.THUMBNAIL_SCALE_FACTOR
    except Exception:
        tsf = 1.0
    return int(round(value * get_scale_factor() * tsf))


def cascade(window, parent=None, step=CASCADE_STEP):
    """
    Place *window* at a fixed offset below-right of *parent* (or of the screen
    origin when no parent is given), clamped to the screen. Every child window
    goes through this, so the window layout is the same on every run.
    """
    try:
        window.update_idletasks()
        offset = scale_value(step)
        if parent is not None and parent.winfo_exists():
            x = parent.winfo_rootx() + offset
            y = parent.winfo_rooty() + offset
        else:
            x = offset
            y = offset
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        x = max(0, min(x, screen_w - offset * 2))
        y = max(0, min(y, screen_h - offset * 2))
        # wm_geometry bypasses CTk geometry scaling (these are real pixels)
        window.wm_geometry(f"+{x}+{y}")
    except Exception:
        pass


def place_main_window(window):
    """Size the main window to its content and place it deterministically.

    Call at the end of building the window's widgets (before mainloop): CTk
    applies a fixed 600x500 default geometry at creation, which would otherwise
    clip wide content and leave empty space below short content.

    Setting the measured content size here — synchronously, before the window
    is first mapped — makes it appear directly at the right size, instead of
    the default size visibly resizing into place a frame later."""
    try:
        window.update_idletasks()
        # winfo_reqwidth/reqheight are in real pixels, but CTk's geometry()
        # works in logical units and multiplies by its window scaling, so divide
        # the requirement by the scale factor.
        scale = get_scale_factor() or 1.0
        logical_w = int(round(window.winfo_reqwidth() / scale))
        logical_h = int(round(window.winfo_reqheight() / scale))
        x = int(window.winfo_screenwidth() * 0.05)
        y = int(window.winfo_screenheight() * 0.07)
        # CTk caps maxsize at its 600x500 default for ~1s after a scaling change,
        # which would clip a wider window — lift the cap to the screen size.
        window.maxsize(int(window.winfo_screenwidth() / scale),
                       int(window.winfo_screenheight() / scale))
        # Pin minsize to the content size: the control row is never clipped, even
        # if the user shrinks the window.
        window.minsize(logical_w, logical_h)
        # Set the explicit size+position once so the window appears at the right
        # size immediately (no default-size flash)...
        window.geometry(f"{logical_w}x{logical_h}+{x}+{y}")
        window.update_idletasks()
        # ...then hand sizing back to the content so the window still grows when
        # the progress bar is added during a run (an explicit WxH alone would
        # freeze it). Re-assert the position, which wm_geometry("") clears.
        window.wm_geometry("")
        window.wm_geometry(f"+{x}+{y}")
        window.update_idletasks()
    except Exception:
        pass


class ThemedProgressBar(ctk.CTkProgressBar):
    """
    CTkProgressBar with a ttk.Progressbar-compatible API so existing call
    sites (config/[]/start/stop with maximum+value semantics) keep working.
    """

    def __init__(self, master, orient="horizontal", length=None,
                 mode="determinate", maximum=100, value=0, **kwargs):
        if length is not None:
            kwargs.setdefault("width", length)
        super().__init__(master, orientation=orient, **kwargs)
        self._maximum = max(1, maximum)
        self._mode = mode
        super().configure(mode=("indeterminate" if mode == "indeterminate" else "determinate"))
        self._set_fraction(value)

    def _set_fraction(self, value):
        try:
            self.set(max(0.0, min(1.0, value / self._maximum)))
        except Exception:
            pass

    def configure(self, require_redraw=False, **kwargs):
        if "orient" in kwargs:
            kwargs["orientation"] = kwargs.pop("orient")
        if "length" in kwargs:
            kwargs["width"] = kwargs.pop("length")
        maximum = kwargs.pop("maximum", None)
        value = kwargs.pop("value", None)
        mode = kwargs.pop("mode", None)
        if mode is not None:
            self._mode = mode
            kwargs["mode"] = mode
            if mode == "determinate":
                try:
                    super().stop()
                except Exception:
                    pass
        if kwargs:
            super().configure(require_redraw=require_redraw, **kwargs)
        if maximum is not None:
            self._maximum = max(1, maximum)
        if value is not None:
            self._set_fraction(value)

    config = configure

    def start(self, interval=None):
        """Begin indeterminate animation (interval accepted for ttk compat)."""
        self._mode = "indeterminate"
        try:
            super().configure(mode="indeterminate")
            super().start()
        except Exception:
            pass

    def stop(self):
        try:
            super().stop()
        except Exception:
            pass

    def __setitem__(self, key, value):
        if key == "value":
            self._set_fraction(value)
        elif key == "maximum":
            self._maximum = max(1, value)
        else:
            self.configure(**{key: value})

    def __getitem__(self, key):
        if key == "value":
            return self.get() * self._maximum
        if key == "maximum":
            return self._maximum
        return self.cget(key)


class TitledFrame(ctk.CTkFrame):
    """CTkFrame with a bold title row — replacement for tk.LabelFrame.

    Add child widgets to ``.content`` (a transparent inner CTkFrame)."""

    def __init__(self, master, title="", **kwargs):
        super().__init__(master, **kwargs)
        self.title_label = ctk.CTkLabel(self, text=title,
                                        font=ctk.CTkFont(size=13, weight="bold"),
                                        anchor="w")
        self.title_label.pack(fill="x", padx=scale_value(12), pady=(scale_value(8), 0))
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True,
                          padx=scale_value(10), pady=scale_value(8))

    def set_title_color(self, color):
        try:
            self.title_label.configure(text_color=color)
        except Exception:
            pass
