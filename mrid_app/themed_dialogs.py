"""
Dark-themed, drop-in replacements for tkinter.messagebox.

Native tkinter.messagebox dialogs are not styled by CustomTkinter, so on a dark
desktop they render unreadable (light/clashing colors). These functions build
modal CustomTkinter dialogs with the app's dark theme while keeping the same
call signatures as tkinter.messagebox, so existing
``messagebox.showinfo(title, message, parent=...)`` calls work unchanged.
"""

import tkinter as tk

import customtkinter as ctk

from . import ui_theme
from .ui_theme import scale_value, cascade
from .logging_config import get_logger

logger = get_logger()


def _resolve_parent(parent):
    if parent is not None:
        return parent
    return getattr(tk, "_default_root", None)


def _dialog(title, message, parent, buttons, default_value, accent):
    """Build a modal themed dialog. *buttons* is a list of
    (label, return_value, fg_color, hover_color). Returns the chosen value, or
    *default_value* if the window is closed without a choice."""
    root = _resolve_parent(parent)
    if root is None:
        # No GUI context (e.g. headless) — nothing to show.
        logger.info(f"{title}: {message}")
        return default_value
    try:
        win = ctk.CTkToplevel(root)
        win.title(ui_theme.t(str(title or "")))
        win.resizable(False, False)
        try:
            win.transient(root)
        except Exception:
            pass

        body = ctk.CTkFrame(win, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=scale_value(20), pady=scale_value(16))

        if title:
            ctk.CTkLabel(
                body, text=ui_theme.t(str(title)),
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=accent, anchor="w", justify="left",
            ).pack(fill="x", pady=(0, scale_value(6)))

        ctk.CTkLabel(
            body, text=ui_theme.t(str(message or "")),
            wraplength=scale_value(440), justify="left", anchor="w",
        ).pack(fill="x", pady=(0, scale_value(14)))

        result = {"v": default_value}

        def _choose(value):
            result["v"] = value
            try:
                win.grab_release()
            except Exception:
                pass
            try:
                win.destroy()
            except Exception:
                pass

        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack()
        first_btn = None
        for label, value, fg, hover in buttons:
            b = ctk.CTkButton(btn_row, text=ui_theme.t(label), width=scale_value(96),
                              fg_color=fg, hover_color=hover,
                              command=(lambda v=value: _choose(v)))
            b.pack(side="left", padx=scale_value(6))
            first_btn = first_btn or b

        win.protocol("WM_DELETE_WINDOW", lambda: _choose(default_value))
        win.update_idletasks()
        cascade(win, root)
        try:
            win.grab_set()
        except Exception:
            pass
        try:
            win.lift()
            if first_btn is not None:
                first_btn.focus_set()
        except Exception:
            pass
        win.wait_window()
        return result["v"]
    except Exception as exc:
        logger.debug(f"Themed dialog failed ({title}): {exc}")
        return default_value


def showinfo(title=None, message=None, parent=None, **kwargs):
    return _dialog(title, message, parent,
                   [("OK", "ok", ui_theme.NEUTRAL, ui_theme.NEUTRAL_HOVER)],
                   "ok", "#4aa3ff")


def showwarning(title=None, message=None, parent=None, **kwargs):
    return _dialog(title, message, parent,
                   [("OK", "ok", ui_theme.YELLOW, ui_theme.YELLOW_HOVER)],
                   "ok", "#e0b020")


def showerror(title=None, message=None, parent=None, **kwargs):
    return _dialog(title, message, parent,
                   [("OK", "ok", ui_theme.RED, ui_theme.RED_HOVER)],
                   "ok", "#ff6b6b")


def askyesno(title=None, message=None, parent=None, **kwargs):
    return bool(_dialog(
        title, message, parent,
        [("Yes", True, ui_theme.GREEN, ui_theme.GREEN_HOVER),
         ("No", False, ui_theme.NEUTRAL, ui_theme.NEUTRAL_HOVER)],
        False, "#4aa3ff"))
