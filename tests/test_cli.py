"""Tests for the ``mrid`` console entry point dispatcher."""
import io
import sys
import types
import contextlib


def _with_fake_module(name, fake):
    """Context manager swapping sys.modules[name] for a fake, then restoring."""
    @contextlib.contextmanager
    def _cm():
        orig = sys.modules.get(name)
        sys.modules[name] = fake
        try:
            yield
        finally:
            if orig is not None:
                sys.modules[name] = orig
            else:
                sys.modules.pop(name, None)
    return _cm()


def test_cli_no_args_launches_gui():
    from mrid_app import cli
    called = {}
    fake = types.ModuleType("run")
    fake.main = lambda *a, **k: called.setdefault("gui", True)
    with _with_fake_module("run", fake):
        cli.main([])
    assert called.get("gui") is True


def test_cli_auto_routes_to_headless_with_rewritten_argv():
    from mrid_app import cli
    captured = {}
    fake = types.ModuleType("headless_autoclassify")
    fake.main = lambda: captured.setdefault("argv", list(sys.argv))
    orig_argv = sys.argv
    with _with_fake_module("headless_autoclassify", fake):
        cli.main(["auto", "-i", "in", "-o", "out", "--model-type", "combined"])
    sys.argv = orig_argv
    assert captured["argv"] == ["mrid auto", "-i", "in", "-o", "out", "--model-type", "combined"]


def test_cli_help_prints_usage_without_launching():
    from mrid_app import cli
    launched = {"gui": False}
    fake = types.ModuleType("run")
    fake.main = lambda *a, **k: launched.__setitem__("gui", True)
    for flag in ("help", "-h", "--help"):
        buf = io.StringIO()
        with _with_fake_module("run", fake), contextlib.redirect_stdout(buf):
            cli.main([flag])
        out = buf.getvalue()
        assert "Usage:" in out and "mrid auto" in out
    assert launched["gui"] is False
