"""Tk objects must release their Tcl resources on the main thread only.

A Tk variable, font or image caught in a reference cycle is freed by Python's
cyclic garbage collector, in whatever thread triggers a collection. Its __del__
calls Tcl, and with a threaded Tcl that call waits for the main thread. If the
main thread is waiting on that thread at the time (Thread.start() waits for the
new thread to begin), the app hangs. ui_theme.guard_tk_finalizers defers such
finalizers to the main thread.

Each scenario runs in a subprocess so a regression shows up as a timeout rather
than hanging the test run.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCENARIO = r'''
import gc, sys, threading, tkinter as tk
sys.path.insert(0, {repo!r})
try:
    root = tk.Tk()
except tk.TclError:
    print("NO_DISPLAY"); sys.exit(0)
root.withdraw()
print("THREADED", root.tk.eval("set tcl_platform(threaded)"), flush=True)
if {guard}:
    from mrid_app import ui_theme
    ui_theme.guard_tk_finalizers(root)
gc.disable()

class Cycle: pass
c = Cycle(); c.me = c
kind = {kind!r}
if kind == "var":
    c.obj = tk.StringVar(root, value="x"); name = c.obj._name
    exists = lambda: root.tk.getboolean(root.tk.call("info", "exists", name))
elif kind == "font":
    import tkinter.font as tkfont
    c.obj = tkfont.Font(root, size=12); name = c.obj.name
    exists = lambda: name in root.tk.splitlist(root.tk.call("font", "names"))
else:
    from PIL import Image, ImageTk
    c.obj = ImageTk.PhotoImage(Image.new("RGB", (4, 4)), master=root); name = str(c.obj)
    exists = lambda: name in root.tk.splitlist(root.tk.call("image", "names"))
del c                                   # cyclic garbage holding a Tk object

# Make the collector fire during the new thread's startup, as an automatic
# collection can on any allocation.
orig = threading.Thread._set_ident
def set_ident(self):
    orig(self)
    gc.collect()
threading.Thread._set_ident = set_ident

def on_click():                         # like a button command starting work
    threading.Thread(target=lambda: None, daemon=True).start()
    print("STARTED", flush=True)
    root.after(1200, check)

def check():
    print("RELEASED", not exists(), flush=True)
    root.destroy()

root.after(200, on_click)
root.mainloop()
'''


def _run(kind, guard, timeout):
    code = _SCENARIO.format(repo=REPO, kind=kind, guard=guard)
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=timeout)
        return proc.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        return (out.decode() if isinstance(out, bytes) else out) + "\nTIMEOUT"


def test_starting_a_thread_does_not_hang_on_tk_garbage():
    for kind in ("var", "font", "pil"):
        out = _run(kind, guard=True, timeout=30)
        if "NO_DISPLAY" in out:
            print("      (skipped: no display)")
            return
        assert "STARTED" in out, f"{kind}: Thread.start() hung:\n{out}"
        assert "RELEASED True" in out, f"{kind}: the Tcl object was never released:\n{out}"


def test_premise_without_the_guard_it_hangs():
    """Confirms the scenario above really exercises the deadlock."""
    out = _run("var", guard=False, timeout=6)
    if "NO_DISPLAY" in out or "THREADED 0" in out:
        print("      (skipped: needs a display and a threaded Tcl)")
        return
    assert "TIMEOUT" in out and "STARTED" not in out, out


def test_main_thread_finalization_is_immediate():
    """On the main thread nothing is deferred."""
    import gc
    import tkinter as tk
    from mrid_app import ui_theme
    try:
        root = tk.Tk()
    except tk.TclError:
        print("      (skipped: no display)")
        return
    try:
        root.withdraw()
        ui_theme.guard_tk_finalizers(root)
        var = tk.StringVar(root, value="x")
        name = var._name
        del var
        gc.collect()
        assert not root.tk.getboolean(root.tk.call("info", "exists", name))
        assert not ui_theme._deferred_finalizers
    finally:
        root.destroy()
