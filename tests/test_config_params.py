"""Missing config settings: detection, validation, writing, and the CLI/GUI prompts.

A config written by an older MRID version lacks settings added since. The GUI
offers to add them; the headless CLI used to refuse to run. Both now go through
mrid_app.config_params, so these tests cover the shared rules and both front ends.
"""

import ast
import contextlib
import io
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

from mrid_app import config
from mrid_app import config_params as cp
from mrid_app.config_io import get_default_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── helpers ──────────────────────────────────────────────────────────────────

def _repo_config_source():
    with open(get_default_config_path(), encoding="utf-8") as fh:
        return fh.read()


def _drop(source, names):
    """*source* with the top-level assignments to *names* removed."""
    lines = source.splitlines(keepends=True)
    doomed = set()
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names):
            doomed.update(range(node.lineno - 1, node.end_lineno))
    return "".join(line for i, line in enumerate(lines) if i not in doomed)


@contextlib.contextmanager
def _old_config(drop=("FUSION_METHOD", "STUDY_ORDER_SEED")):
    """A temporary project config lacking *drop*, as an older MRID wrote it."""
    tmp = tempfile.mkdtemp(prefix="mrid_cfg_")
    path = os.path.join(tmp, "config.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_drop(_repo_config_source(), set(drop)))
    try:
        yield path
    finally:
        os.chmod(tmp, 0o755)
        shutil.rmtree(tmp, ignore_errors=True)


@contextlib.contextmanager
def _preserve_config_module():
    """Restore mrid_app.config after a test that loads a project config into it."""
    saved = {n: getattr(config, n) for n in dir(config) if n.isupper()}
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(config, n, v)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _scripted(*answers):
    """An input() replacement returning *answers* in order, then EOF."""
    it = iter(answers)

    def fake_input(*_a):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return fake_input


def _load_module(path):
    return cp.load_config_module(path)


def _temp_files(path):
    """Leftover temp files from an atomic write next to *path*."""
    return [f for f in os.listdir(os.path.dirname(path)) if f.startswith(".config.")]


# ── detection ────────────────────────────────────────────────────────────────

def test_find_missing_reports_dropped_settings_in_repository_order():
    with _old_config(drop=("STUDY_ORDER_SEED", "FUSION_METHOD")) as path:
        assert cp.find_missing_params(path) == ["FUSION_METHOD", "STUDY_ORDER_SEED"]


def test_complete_config_has_nothing_missing():
    assert cp.find_missing_params(get_default_config_path()) == []


def test_old_configs_with_the_removed_cache_slot_still_load():
    # SEMANTIC_MODEL was an in-memory cache, not a setting; configs written
    # before it was removed still assign it, which must be harmless.
    with _old_config(drop=()) as path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\nSEMANTIC_MODEL = None  # always None initially\n")
        assert cp.find_missing_params(path) == []
    assert "SEMANTIC_MODEL" not in cp.repository_defaults()


def test_find_missing_raises_on_a_broken_file():
    with _old_config() as path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\nTHIS IS NOT PYTHON\n")
        try:
            cp.find_missing_params(path)
        except SyntaxError:
            return
        raise AssertionError("a syntax error must propagate, not read as 'missing'")


def test_repository_defaults_ignore_a_previously_loaded_project():
    # Loading a project config overwrites the config module; defaults must not
    # come from there, or an old project would inherit the previous project's value.
    with _preserve_config_module():
        config.FUSION_METHOD = "max"
        config.OVERRIDE_THRESHOLD = 0.9
        defaults = cp.repository_defaults()
    assert defaults["FUSION_METHOD"] == "reliability"
    assert defaults["OVERRIDE_THRESHOLD"] == 0.5


def _set_labels(path, labels):
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    from mrid_app.config_io import replace_single_assignment
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(replace_single_assignment(source, "TARGET_SERIES", repr(labels)))


def test_prefer_3d_default_follows_the_projects_own_labels():
    # The repository PREFER_3D lists the example labels (T1, T2, FLAIR, T1c).
    # Suggesting that to a project with other labels would list labels it lacks.
    with _old_config(drop=("PREFER_3D",)) as path:
        _set_labels(path, ["AX DCE", "T1", "Other"])
        suggested = cp.project_defaults(path)["PREFER_3D"]
        assert suggested == {"AX DCE": False, "T1": True, "Other": False}
        # Same behavior MRID gives the repository dict (it reads .get(label, False)).
        repo = cp.repository_defaults()["PREFER_3D"]
        assert all(suggested[l] == repo.get(l, False) for l in suggested)


def test_prefer_3d_default_follows_labels_entered_in_the_same_prompt():
    with _old_config(drop=("TARGET_SERIES", "PREFER_3D")) as path:
        out = io.StringIO()
        result = cp.prompt_for_missing_params(
            path, ["TARGET_SERIES", "PREFER_3D"],
            input_fn=_scripted("e", "T1, DWI, Other", "a", ""), out=out)
        assert result == {"TARGET_SERIES": ["T1", "DWI", "Other"],
                          "PREFER_3D": {"T1": True, "DWI": False, "Other": False}}
        assert '"DWI": False' in out.getvalue().split("[2/2]")[1]


def test_accepted_defaults_are_written_for_the_project():
    with _old_config(drop=("PREFER_3D",)) as path:
        _set_labels(path, ["SAG T2", "Other"])
        cp.resolve_missing_params_cli(path, ["PREFER_3D"], interactive=False,
                                      accept_defaults=True, out=io.StringIO())
        assert _load_module(path).PREFER_3D == {"SAG T2": False, "Other": False}


# ── validation ───────────────────────────────────────────────────────────────

def _ok(name, value, expected=None):
    got = cp.validate_value(name, value)
    want = value if expected is None else expected
    assert got == want and type(got) is type(want), f"{name}: {value!r} -> {got!r}, want {want!r}"


def _bad(name, value):
    try:
        cp.validate_value(name, value)
    except ValueError as e:
        assert str(e), "rejections must explain themselves"
        return
    raise AssertionError(f"{name} accepted invalid {value!r}")


def test_every_repository_default_is_valid_and_round_trips():
    for name, value in cp.repository_defaults().items():
        _ok(name, value, value)
        assert ast.literal_eval(cp.format_value(name, value)) == value, name


def test_choice_settings_are_case_insensitive_and_canonicalized():
    _ok("THUMBNAIL_FORMAT", "png", "PNG")
    _ok("FUSION_METHOD", "Average", "average")
    _ok("LOG_LEVEL", "debug", "DEBUG")
    _ok("ANALYZE_HDBSCAN_CLUSTER_METHOD", "EOM", "eom")
    _bad("THUMBNAIL_FORMAT", "GIF")
    _bad("FUSION_METHOD", "mean")
    _bad("FUSION_METHOD", 1)


def test_numeric_ranges():
    _ok("THUMBNAIL_QUALITY", 1)
    _ok("THUMBNAIL_QUALITY", 100)
    for v in (0, 101, True, 50.0, "50"):
        _bad("THUMBNAIL_QUALITY", v)
    _ok("OVERRIDE_THRESHOLD", 1, 1.0)
    _ok("OVERRIDE_THRESHOLD", 0.2)
    for v in (-0.1, 1.5, "x", False):
        _bad("OVERRIDE_THRESHOLD", v)
    _ok("STUDY_ORDER_SEED", -7)
    _bad("STUDY_ORDER_SEED", 4.2)
    _bad("NUM_PARALLEL_WORKERS", 0)
    _bad("IMAGE_SIZE", 8)


def test_booleans_accept_common_words():
    _ok("RANDOMIZE_STUDY_ORDER", "yes", True)
    _ok("RANDOMIZE_STUDY_ORDER", "off", False)
    _ok("RANDOMIZE_STUDY_ORDER", 1, True)
    _bad("RANDOMIZE_STUDY_ORDER", 2)
    _bad("RANDOMIZE_STUDY_ORDER", "maybe")


def test_target_series():
    _ok("TARGET_SERIES", "T1, T2 FS, Other", ["T1", "T2 FS", "Other"])
    _ok("TARGET_SERIES", ("T1", "Other"), ["T1", "Other"])
    _bad("TARGET_SERIES", [])
    _bad("TARGET_SERIES", ["T1", "T1"])
    _bad("TARGET_SERIES", ["T1/post"])  # would be silently altered by the sanitizer
    _bad("TARGET_SERIES", ["T1", 3])


def test_prefer_3d_bool_or_dict():
    _ok("PREFER_3D", True)
    _ok("PREFER_3D", {"T1": "yes", "Other": False}, {"T1": True, "Other": False})
    _bad("PREFER_3D", {"T1": 3})
    _bad("PREFER_3D", [True])


def test_study_name_template_fields():
    _ok("STUDY_NAME_TEMPLATE", "{mrn}_{study_uid}")
    _bad("STUDY_NAME_TEMPLATE", "{patient}")
    _bad("STUDY_NAME_TEMPLATE", "same-for-every-study")
    _bad("STUDY_NAME_TEMPLATE", "{mrn")
    _bad("STUDY_NAME_TEMPLATE", "")


def test_dicom_header_fields():
    _ok("DICOM_HEADER_FIELDS", [["TR", "RepetitionTime"]], [("TR", "RepetitionTime")])
    _bad("DICOM_HEADER_FIELDS", [])
    _bad("DICOM_HEADER_FIELDS", [("TR",)])
    _bad("DICOM_HEADER_FIELDS", [("TR", "")])


def test_settings_without_an_explicit_rule_are_type_checked_by_default():
    assert cp._rule_for("FUTURE_INT", 3).check(4) == 4
    try:
        cp._rule_for("FUTURE_INT", 3).check("4")
    except ValueError:
        pass
    else:
        raise AssertionError("an int setting must reject text")
    assert cp._rule_for("FUTURE_LIST", [1]).check([2]) == [2]


def test_parse_value_accepts_literals_and_plain_text():
    assert cp.parse_value("FUSION_METHOD", "'product'") == "product"
    assert cp.parse_value("FUSION_METHOD", "product") == "product"
    assert cp.parse_value("ALLOW_DUPLICATE_LABELS", "true") is True
    assert cp.parse_value("EXCLUDE_TYPES", "DERIVED, MPR") == ["DERIVED", "MPR"]
    assert cp.parse_value("NUM_EPOCHS", " 50 ") == 50
    for name, text in (("NUM_EPOCHS", ""), ("NUM_EPOCHS", "fifty"),
                       ("INCLUDE_TYPES", "[1, 2]"), ("FUSION_METHOD", "__import__('os')")):
        try:
            cp.parse_value(name, text)
        except ValueError:
            continue
        raise AssertionError(f"{name} accepted {text!r}")


def test_every_setting_has_a_description():
    descriptions = cp.param_descriptions()
    undocumented = [n for n in cp.repository_defaults() if not descriptions.get(n)]
    assert not undocumented, f"add a comment in config.py for: {undocumented}"


# ── writing ──────────────────────────────────────────────────────────────────

def test_add_params_appends_without_touching_existing_content():
    with _old_config() as path:
        before = _read(path)
        os.chmod(path, 0o640)
        cp.add_params_to_config(path, {"FUSION_METHOD": "product", "STUDY_ORDER_SEED": 7})
        after = _read(path)

        assert after.startswith(before.rstrip("\n")), "existing content must be unchanged"
        assert "Added by MRID on" in after
        assert "FUSION_METHOD = 'product'" in after and "STUDY_ORDER_SEED = 7" in after
        # Descriptions travel with the setting, so the file documents itself.
        assert "# How the header and image model probabilities are combined" in after
        module = _load_module(path)
        assert module.FUSION_METHOD == "product" and module.STUDY_ORDER_SEED == 7
        assert cp.find_missing_params(path) == []
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o640, "file mode must be kept"
        assert _temp_files(path) == [], "no temp files left"


def test_add_params_writes_multiline_settings_readably():
    with _old_config(drop=("PREFER_3D", "DICOM_HEADER_FIELDS")) as path:
        values = {"PREFER_3D": {"T1": True, "Other": False},
                  "DICOM_HEADER_FIELDS": [("TR", "RepetitionTime")]}
        cp.add_params_to_config(path, values)
        module = _load_module(path)
        assert module.PREFER_3D == values["PREFER_3D"]
        assert module.DICOM_HEADER_FIELDS == values["DICOM_HEADER_FIELDS"]
        assert '    "T1": True,\n' in _read(path)


def test_add_params_writes_through_a_symlink():
    with _old_config() as path:
        link = os.path.join(os.path.dirname(path), "linked_config.py")
        os.symlink(path, link)
        cp.add_params_to_config(link, {"FUSION_METHOD": "max"})
        assert os.path.islink(link), "the symlink must survive"
        assert _load_module(path).FUSION_METHOD == "max"


def test_add_params_leaves_file_untouched_on_failure():
    with _old_config() as path:
        before = _read(path)
        try:
            cp.add_params_to_config(path, {"FUSION_METHOD": object()})  # not a literal
        except Exception:
            pass
        else:
            raise AssertionError("a value that cannot be written must raise")
        assert _read(path) == before
        assert _temp_files(path) == []


# ── terminal prompt ──────────────────────────────────────────────────────────

MISSING = ["FUSION_METHOD", "STUDY_ORDER_SEED"]


def _prompt(*answers):
    out = io.StringIO()
    result = cp.prompt_for_missing_params("/p/config.py", MISSING,
                                          input_fn=_scripted(*answers), out=out)
    return result, out.getvalue()


def test_prompt_enter_accepts_defaults():
    result, text = _prompt("", "", "")
    assert result == {"FUSION_METHOD": "reliability", "STUDY_ORDER_SEED": 42}
    # The user sees what each setting does, what is valid, and the default.
    assert "How the header and image model probabilities are combined" in text
    assert "Valid values: one of 'reliability'" in text
    assert "Default: 'reliability'" in text
    assert "FUSION_METHOD = 'reliability'" in text  # the confirmation summary


def test_prompt_enter_value_is_validated_until_valid():
    result, text = _prompt("e", "mean", "Product", "a", "y")
    assert result == {"FUSION_METHOD": "product", "STUDY_ORDER_SEED": 42}
    assert "Invalid value: FUSION_METHOD must be one of" in text


def test_prompt_blank_value_goes_back_to_the_choice():
    result, _ = _prompt("e", "", "a", "e", "123", "")
    assert result == {"FUSION_METHOD": "reliability", "STUDY_ORDER_SEED": 123}


def test_prompt_remaining_accepts_all_defaults():
    result, _ = _prompt("r", "")
    assert result == {"FUSION_METHOD": "reliability", "STUDY_ORDER_SEED": 42}


def test_prompt_remaining_is_not_offered_for_the_last_setting():
    _, text = _prompt("a", "r", "a", "")
    last = text.split("[2/2]")[1]
    assert "[R]emaining" not in last
    assert "Please answer" in last


def test_prompt_cancel_eof_and_declined_confirmation_return_none():
    assert _prompt("c")[0] is None
    assert _prompt()[0] is None                  # EOF / Ctrl-D
    # 'c' typed as a *value* is not a cancel, just an invalid value.
    result, text = _prompt("e", "c")
    assert result is None and "Invalid value: FUSION_METHOD" in text
    assert _prompt("", "", "n")[0] is None       # declined at the confirmation


def test_prompt_unknown_answer_asks_again():
    result, text = _prompt("x", "a", "a", "")
    assert result is not None and "Please answer" in text


# ── CLI resolution ───────────────────────────────────────────────────────────

def test_cli_without_a_terminal_stops_with_copyable_instructions():
    with _old_config() as path:
        before = _read(path)
        out = io.StringIO()
        assert cp.resolve_missing_params_cli(path, MISSING, interactive=False, out=out) is None
        text = out.getvalue()
        assert _read(path) == before
        assert "Missing setting(s): FUSION_METHOD, STUDY_ORDER_SEED" in text
        assert "FUSION_METHOD = 'reliability'\nSTUDY_ORDER_SEED = 42\n" in text
        assert "--accept-config-defaults" in text


def test_cli_instructions_paste_into_a_working_config():
    with _old_config() as path:
        out = io.StringIO()
        cp.resolve_missing_params_cli(path, MISSING, interactive=False, out=out)
        pasted = out.getvalue().split("if needed:\n\n", 1)[1].split("\nOr re-run", 1)[0]
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n" + pasted)
        assert cp.find_missing_params(path) == []


def test_cli_accept_defaults_writes_the_file():
    with _old_config() as path:
        values = cp.resolve_missing_params_cli(path, MISSING, interactive=False,
                                               accept_defaults=True, out=io.StringIO())
        assert values == {"FUSION_METHOD": "reliability", "STUDY_ORDER_SEED": 42}
        assert cp.find_missing_params(path) == []


def test_cli_interactive_cancel_changes_nothing():
    with _old_config() as path:
        before = _read(path)
        out = io.StringIO()
        assert cp.resolve_missing_params_cli(path, MISSING, interactive=True,
                                             input_fn=_scripted("c"), out=out) is None
        assert _read(path) == before
        assert "Cancelled. The config file was not changed." in out.getvalue()
        assert "FUSION_METHOD = 'reliability'" in out.getvalue()


@contextlib.contextmanager
def _read_only_dir(path):
    folder = os.path.dirname(path)
    os.chmod(folder, 0o555)
    try:
        yield
    finally:
        os.chmod(folder, 0o755)


def _can_block_writes():
    return hasattr(os, "geteuid") and os.geteuid() != 0


def test_cli_write_failure_can_continue_for_this_run_only():
    if not _can_block_writes():
        return
    with _old_config() as path:
        before = _read(path)
        with _read_only_dir(path):
            out = io.StringIO()
            values = cp.resolve_missing_params_cli(
                path, MISSING, interactive=True, input_fn=_scripted("e", "max", "", "", "y"),
                out=out)
            assert values == {"FUSION_METHOD": "max", "STUDY_ORDER_SEED": 42}
            assert "Could not write to" in out.getvalue()
            assert "FUSION_METHOD = 'max'" in out.getvalue(), "instructions keep the chosen value"

            declined = cp.resolve_missing_params_cli(
                path, MISSING, interactive=True, input_fn=_scripted("", "", "", "n"),
                out=io.StringIO())
            assert declined is None

            batch = cp.resolve_missing_params_cli(path, MISSING, interactive=False,
                                                  accept_defaults=True, out=io.StringIO())
            assert batch == {"FUSION_METHOD": "reliability", "STUDY_ORDER_SEED": 42}
        assert _read(path) == before


# ── headless loader ──────────────────────────────────────────────────────────

def _headless():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import headless_autoclassify
    return headless_autoclassify


def test_headless_accept_defaults_loads_and_updates_the_file():
    h = _headless()
    with _old_config() as path, _preserve_config_module():
        config.FUSION_METHOD = "max"  # left over from a previously loaded project
        assert h.load_custom_config(path, accept_defaults=True, out=io.StringIO())
        assert config.FUSION_METHOD == "reliability"
        assert cp.find_missing_params(path) == []


def test_headless_non_interactive_refuses_without_the_flag():
    h = _headless()
    with _old_config() as path, _preserve_config_module():
        before = _read(path)
        assert not h.load_custom_config(path, interactive=False, out=io.StringIO())
        assert _read(path) == before


def test_headless_interactive_value_is_saved_and_applied():
    h = _headless()
    with _old_config() as path, _preserve_config_module():
        ok = h.load_custom_config(path, interactive=True,
                                  input_fn=_scripted("e", "product", "e", "2024", ""),
                                  out=io.StringIO())
        assert ok
        assert config.FUSION_METHOD == "product" and config.STUDY_ORDER_SEED == 2024
        module = _load_module(path)
        assert module.FUSION_METHOD == "product" and module.STUDY_ORDER_SEED == 2024


def test_headless_unsaved_values_still_apply_to_the_run():
    if not _can_block_writes():
        return
    h = _headless()
    with _old_config() as path, _preserve_config_module():
        with _read_only_dir(path):
            ok = h.load_custom_config(path, interactive=True,
                                      input_fn=_scripted("e", "average", "", "", "y"),
                                      out=io.StringIO())
        assert ok and config.FUSION_METHOD == "average"
        assert cp.find_missing_params(path) == MISSING


def test_headless_complete_and_broken_configs():
    h = _headless()
    with _preserve_config_module():
        assert h.load_custom_config(get_default_config_path(), interactive=False)
    with _old_config() as path, _preserve_config_module():
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\nBROKEN = (\n")
        assert not h.load_custom_config(path, accept_defaults=True, out=io.StringIO())
    assert not h.load_custom_config("/no/such/config.py")


# ── end to end: the real command ─────────────────────────────────────────────

def _run_headless(config_path, *extra, stdin=subprocess.DEVNULL):
    tmp = os.path.dirname(config_path)
    inp, outp = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    os.makedirs(inp, exist_ok=True)
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "headless_autoclassify.py"),
         "-i", inp, "-o", outp, "--config", config_path, *extra],
        stdin=stdin, capture_output=True, text=True, cwd=REPO, timeout=300)


def test_headless_command_without_terminal_exits_with_instructions():
    with _old_config() as path:
        before = _read(path)
        proc = _run_headless(path)
        assert proc.returncode == 1, proc.stderr[-2000:]
        assert "FUSION_METHOD = 'reliability'" in proc.stderr
        assert "--accept-config-defaults" in proc.stderr
        assert _read(path) == before


def test_headless_command_accept_defaults_runs():
    with _old_config() as path:
        proc = _run_headless(path, "--accept-config-defaults")
        # An empty input folder: the run gets past config loading and finds nothing.
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert cp.find_missing_params(path) == []


def test_headless_command_in_a_real_terminal():
    """Drive the actual prompt through a pseudo-terminal, as a user would."""
    try:
        import pty
        import select
    except ImportError:  # Windows
        return
    with _old_config() as path:
        tmp = os.path.dirname(path)
        os.makedirs(os.path.join(tmp, "in"), exist_ok=True)
        master, slave = pty.openpty()
        proc = subprocess.Popen(
            [sys.executable, os.path.join(REPO, "headless_autoclassify.py"),
             "-i", os.path.join(tmp, "in"), "-o", os.path.join(tmp, "out"), "--config", path],
            stdin=slave, stdout=slave, stderr=slave, cwd=REPO, close_fds=True)
        os.close(slave)
        seen = ""

        def expect(text, timeout=120):
            nonlocal seen
            deadline = time.time() + timeout
            while text not in seen:
                if time.time() > deadline:
                    raise AssertionError(f"timed out waiting for {text!r}; got:\n{seen[-1500:]}")
                ready, _, _ = select.select([master], [], [], 0.5)
                if ready:
                    try:
                        seen += os.read(master, 4096).decode(errors="replace")
                    except OSError:
                        break

        try:
            script = [("[1/2] FUSION_METHOD", None), ("(default A): ", "e\n"),
                      ("New value for FUSION_METHOD", "bogus\n"),
                      ("Invalid value", None), ("New value for FUSION_METHOD", "product\n"),
                      ("[2/2] STUDY_ORDER_SEED", None), ("(default A): ", "\n"),
                      ("Write them to the config file? [Y/n]: ", "\n")]
            for text, reply in script:
                expect(text)
                if reply:
                    seen = seen[seen.index(text) + len(text):]  # consume the prompt
                    os.write(master, reply.encode())
            proc.wait(timeout=300)
        finally:
            if proc.poll() is None:
                proc.kill()
            os.close(master)
        assert proc.returncode == 0, seen[-2000:]
        module = _load_module(path)
        assert module.FUSION_METHOD == "product" and module.STUDY_ORDER_SEED == 42


# ── GUI ──────────────────────────────────────────────────────────────────────

def _gui_app_stub(master=None):
    from mrid_app import gui
    app = gui.MainApp.__new__(gui.MainApp)
    app.master = master
    app.active_config_path = None
    return gui, app


def test_gui_load_uses_repository_defaults_for_missing_settings():
    gui, app = _gui_app_stub()
    prompted = []
    app._prompt_add_missing_params = lambda p, m: prompted.append(m)
    with _old_config() as path, _preserve_config_module():
        config.FUSION_METHOD = "max"  # from a previously loaded project
        config.STUDY_ORDER_SEED = 999
        assert gui.MainApp._load_config_from_file(app, path)
        assert config.FUSION_METHOD == "reliability"
        assert config.STUDY_ORDER_SEED == 42
        assert prompted == [MISSING]


def test_gui_dialog_validates_and_writes():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:  # no display
        return
    root.withdraw()
    gui, app = _gui_app_stub(master=root)
    warnings, infos = [], []
    orig = (gui.messagebox.showwarning, gui.messagebox.showinfo, gui.messagebox.showerror)
    gui.messagebox.showwarning = lambda *a, **k: warnings.append(a)
    gui.messagebox.showinfo = lambda *a, **k: infos.append(a)
    gui.messagebox.showerror = lambda *a, **k: warnings.append(a)
    try:
        with _old_config() as path, _preserve_config_module():
            def find(widget, kind):
                found = [widget] if isinstance(widget, kind) else []
                for child in widget.winfo_children():
                    found += find(child, kind)
                return found

            def drive():
                dialog = [w for w in root.winfo_children()
                          if isinstance(w, gui.ctk.CTkToplevel)][0]
                entries = find(dialog, gui.ctk.CTkEntry)
                add = [b for b in find(dialog, gui.ctk.CTkButton)
                       if b.cget("text") == "Add to Config"][0]
                entries[0].delete(0, "end")
                entries[0].insert(0, "mean")          # invalid: must not be written
                add.invoke()
                assert warnings and "FUSION_METHOD" in warnings[-1][1]
                assert cp.find_missing_params(path) == MISSING
                entries[0].delete(0, "end")
                entries[0].insert(0, "product")       # plain text is fine
                add.invoke()

            root.after(300, drive)
            gui.MainApp._prompt_add_missing_params(app, path, MISSING)
            assert infos, "success must be confirmed"
            module = _load_module(path)
            assert module.FUSION_METHOD == "product" and module.STUDY_ORDER_SEED == 42
            assert config.FUSION_METHOD == "product", "the new values must take effect"
    finally:
        (gui.messagebox.showwarning, gui.messagebox.showinfo, gui.messagebox.showerror) = orig
        root.destroy()
