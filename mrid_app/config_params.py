"""Detect, validate and add settings that a project ``config.py`` is missing.

Configs written by older MRID versions lack settings introduced since. This
module is the single place that decides which settings count as missing, what a
valid value for each one is, and how new settings are written back to the file,
so the GUI and the headless CLI treat an old config the same way.
"""

import ast
import datetime
import importlib.util
import os
import re
import shutil
import string
import sys
import tempfile

from .config_io import format_prefer3d_dict, get_default_config_path
from .logging_config import get_logger

logger = get_logger()

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_TEMPLATE_FIELDS = ("mrn", "study_uid", "accn")
_TRUE_WORDS = ("true", "yes", "y", "on", "1")
_FALSE_WORDS = ("false", "no", "n", "off", "0")


# ── Loading ──────────────────────────────────────────────────────────────────

def load_config_module(config_path):
    """Execute *config_path* as a module and return it.

    Raises whatever loading raises (``SyntaxError``, ``NameError``, ...), so the
    caller can report a broken file distinctly from an incomplete one.
    """
    spec = importlib.util.spec_from_file_location("mrid_project_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_param_name(name):
    return name.isupper() and not name.startswith("_")


def repository_defaults():
    """Every setting and its default, read fresh from the repository config.py.

    Read from the file rather than the imported ``mrid_app.config`` module,
    because loading a project config overwrites that module's attributes: after
    one project has been loaded, the module no longer holds the defaults.
    Ordered as the settings appear in the file.
    """
    from . import config as config_module
    try:
        module = load_config_module(get_default_config_path())
        defaults = {n: getattr(module, n) for n in vars(module) if _is_param_name(n)}
    except Exception as e:  # pragma: no cover - the repository config always loads
        logger.warning(f"Could not read the repository config defaults: {e}")
        defaults = {}
    # Anything the running code defines that the file lacks (normally nothing).
    for name in dir(config_module):
        if _is_param_name(name) and name not in defaults:
            defaults[name] = getattr(config_module, name)
    return defaults


def _adapt_prefer_3d(defaults, labels):
    """Key the PREFER_3D default to *labels* (a project's TARGET_SERIES).

    The repository default is keyed to its own example labels. MRID reads
    PREFER_3D with ``.get(label, False)``, so for a project with other labels
    that dict means "no 3D preference" while listing labels the project doesn't
    have. The same behavior keyed to the project's labels is what a user can
    actually read and edit.
    """
    repo = defaults.get("PREFER_3D")
    if isinstance(repo, dict) and isinstance(labels, (list, tuple)) and labels:
        defaults["PREFER_3D"] = {label: bool(repo.get(label, False))
                                 for label in labels if isinstance(label, str)}
    return defaults


def project_defaults(config_path):
    """Defaults to suggest for settings *config_path* lacks: the repository
    defaults, with any that depend on the project's other settings adapted to
    them (currently PREFER_3D, which is keyed by the project's TARGET_SERIES)."""
    defaults = repository_defaults()
    try:
        labels = getattr(load_config_module(config_path), "TARGET_SERIES", None)
    except Exception:
        labels = None
    return _adapt_prefer_3d(defaults, labels)


def find_missing_params(config_path):
    """Names of the settings *config_path* does not define, in repository order.

    Raises if the file cannot be executed (see :func:`load_config_module`).
    """
    module = load_config_module(config_path)
    return [name for name in repository_defaults() if not hasattr(module, name)]


# ── Descriptions ─────────────────────────────────────────────────────────────

def param_descriptions():
    """``{NAME: description}`` taken from the comments in the repository config.

    A trailing ``# comment`` on the assignment wins; otherwise the block of
    comment lines directly above it is used. Section banners are skipped. Keeping
    the text in config.py means there is one description per setting to maintain.
    """
    try:
        with open(get_default_config_path(), "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
    except Exception:  # pragma: no cover
        return {}
    lines = source.splitlines()
    descriptions = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        trailing = lines[node.end_lineno - 1][node.end_col_offset:].strip()
        if trailing.startswith("#"):
            descriptions[name] = trailing.lstrip("#").strip()
            continue
        block = []
        i = node.lineno - 2
        while i >= 0:
            text = lines[i].strip()
            if not text.startswith("#"):
                break
            body = text[1:]
            if body.strip() and set(body.strip()) <= set("=-"):
                break  # section banner
            block.append(body[1:] if body.startswith(" ") else body)
            i -= 1
        if block:
            descriptions[name] = "\n".join(reversed(block)).rstrip()
    return descriptions


# ── Validation ───────────────────────────────────────────────────────────────

class _Rule:
    """A validator plus a short human-readable description of valid values."""

    def __init__(self, check, hint):
        self.check = check
        self.hint = hint


def _type_name(value):
    return type(value).__name__


def _int_rule(lo=None, hi=None):
    if lo is not None and hi is not None:
        hint = f"a whole number from {lo} to {hi}"
    elif lo is not None:
        hint = f"a whole number, {lo} or more"
    else:
        hint = "a whole number"

    def check(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"must be {hint} (got {value!r})")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(f"must be {hint} (got {value})")
        return value
    return _Rule(check, hint)


def _float_rule(lo=None, hi=None):
    if lo is not None and hi is not None:
        hint = f"a number from {lo} to {hi}"
    elif lo is not None:
        hint = f"a number, {lo} or more"
    else:
        hint = "a number"

    def check(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"must be {hint} (got {value!r})")
        value = float(value)
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(f"must be {hint} (got {value})")
        return value
    return _Rule(check, hint)


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ValueError(f"must be True or False (got {value!r})")


def _bool_rule():
    return _Rule(_coerce_bool, "True or False")


def _choice_rule(options):
    options = tuple(options)
    hint = "one of " + ", ".join(repr(o) for o in options)

    def check(value):
        if isinstance(value, str):
            for option in options:
                if value.strip().lower() == option.lower():
                    return option
        raise ValueError(f"must be {hint} (got {value!r})")
    return _Rule(check, hint)


def _text_rule(nonempty=True):
    hint = "non-empty text" if nonempty else "text"

    def check(value):
        if not isinstance(value, str):
            raise ValueError(f"must be {hint} (got {value!r})")
        if nonempty and not value.strip():
            raise ValueError("must not be empty")
        return value
    return _Rule(check, hint)


def _as_str_list(value, name_for_error):
    """Accept a list/tuple of strings, or comma-separated text."""
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"must be a list of text values (got {value!r})")
    items = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"every {name_for_error} must be non-empty text (got {item!r})")
        items.append(item.strip())
    return items


def _str_list_rule(nonempty=False, unique=False):
    hint = "a list of text values, e.g. ['A', 'B'] or A, B"
    if nonempty:
        hint += " (at least one)"

    def check(value):
        items = _as_str_list(value, "entry")
        if nonempty and not items:
            raise ValueError("must contain at least one entry")
        if unique:
            seen = set()
            for item in items:
                if item in seen:
                    raise ValueError(f"contains {item!r} more than once")
                seen.add(item)
        return items
    return _Rule(check, hint)


def _target_series_rule():
    base = _str_list_rule(nonempty=True, unique=True)
    hint = ("a list of labels using letters, digits, spaces, - and _, "
            "e.g. ['T1', 'T2', 'Other'] or T1, T2, Other")

    def check(value):
        labels = base.check(value)
        bad = [label for label in labels if not re.fullmatch(r"[A-Za-z0-9 _\-]+", label)]
        if bad:
            raise ValueError(f"labels may only use letters, digits, spaces, - and _ "
                             f"(check {', '.join(repr(b) for b in bad)})")
        return labels
    return _Rule(check, hint)


def _template_rule():
    allowed = ", ".join("{" + f + "}" for f in _TEMPLATE_FIELDS)
    hint = f"text using any of {allowed}"

    def check(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"must be {hint} (got {value!r})")
        try:
            fields = [f for _, f, _, _ in string.Formatter().parse(value) if f is not None]
        except ValueError as e:
            raise ValueError(f"is not a valid template: {e}")
        unknown = [f for f in fields if f not in _TEMPLATE_FIELDS]
        if unknown:
            raise ValueError(f"uses unknown field(s) {', '.join('{' + f + '}' for f in unknown)}; "
                             f"allowed: {allowed}")
        if not fields:
            raise ValueError(f"must include at least one of {allowed}, or every "
                             "study would get the same folder name")
        return value
    return _Rule(check, hint)


def _prefer_3d_rule():
    hint = "True/False for all labels, or a dict such as {'T1': True, 'Other': False}"

    def check(value):
        if isinstance(value, dict):
            out = {}
            for key, flag in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"keys must be label names (got {key!r})")
                try:
                    out[key] = _coerce_bool(flag)
                except ValueError:
                    raise ValueError(f"value for {key!r} must be True or False (got {flag!r})")
            return out
        try:
            return _coerce_bool(value)
        except ValueError:
            raise ValueError(f"must be {hint} (got {value!r})")
    return _Rule(check, hint)


def _header_fields_rule():
    hint = "a list of (display name, DICOM attribute) pairs"

    def check(value):
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"must be {hint}, with at least one entry")
        out = []
        for pair in value:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not all(isinstance(p, str) and p.strip() for p in pair)):
                raise ValueError(f"each entry must be a (display name, DICOM attribute) "
                                 f"pair of text (got {pair!r})")
            out.append((pair[0], pair[1]))
        return out
    return _Rule(check, hint)


def _rules():
    from .fusion import FUSION_METHODS
    return {
        "STUDY_NAME_TEMPLATE": _template_rule(),
        "THUMBNAIL_FORMAT": _choice_rule(("JPEG", "PNG")),
        "THUMBNAIL_QUALITY": _int_rule(1, 100),
        "TARGET_SERIES": _target_series_rule(),
        "PREFER_3D": _prefer_3d_rule(),
        "INCLUDE_TYPES": _str_list_rule(),
        "EXCLUDE_TYPES": _str_list_rule(),
        "IGNORE_MULTIPLE_ORIENTATIONS": _bool_rule(),
        "SEPARATE_STACKED_SERIES": _bool_rule(),
        "MAX_SPLIT": _int_rule(1),
        "ANALYZE_COVERAGE_THRESHOLD": _float_rule(0.0, 1.0),
        "ANALYZE_HDBSCAN_CLUSTER_METHOD": _choice_rule(("eom", "leaf")),
        "NUM_COLUMNS": _int_rule(1),
        "NUM_ROWS": _int_rule(1),
        "THUMBNAIL_SCALE_FACTOR": _float_rule(0.1),
        "OVERRIDE_THRESHOLD": _float_rule(0.0, 1.0),
        "FUSION_METHOD": _choice_rule(FUSION_METHODS),
        "ALLOW_DUPLICATE_LABELS": _bool_rule(),
        "NUM_PARALLEL_WORKERS": _int_rule(1),
        "RANDOMIZE_STUDY_ORDER": _bool_rule(),
        "STUDY_ORDER_SEED": _int_rule(),
        "IMAGE_SIZE": _int_rule(16),
        "NUM_EPOCHS": _int_rule(1),
        "BATCH_SIZE": _int_rule(1),
        "LR": _float_rule(0.0),
        "CNN_NUM_SLICES": _int_rule(1),
        "N_ESTIMATORS": _int_rule(1),
        "DICOM_HEADER_FIELDS": _header_fields_rule(),
        "CATEGORICAL_UNIQUE_THRESHOLD": _int_rule(1),
        "TEXT_MIN_WORDS": _int_rule(1),
        "ARRAY_SIZE_THRESHOLD": _int_rule(0),
        "SEMANTIC_MODEL_NAME": _text_rule(),
        "LOG_LEVEL": _choice_rule(_LOG_LEVELS),
    }


def _rule_for(name, default=None):
    """The explicit rule for *name*, or one inferred from its default's type so
    that a setting added without a rule here is still type-checked."""
    rule = _rules().get(name)
    if rule is not None:
        return rule
    if isinstance(default, bool):
        return _bool_rule()
    if isinstance(default, int):
        return _int_rule()
    if isinstance(default, float):
        return _float_rule()
    if isinstance(default, str):
        return _text_rule(nonempty=False)
    if isinstance(default, (list, tuple, dict)):
        kind = type(default)

        def check(value):
            if not isinstance(value, kind):
                raise ValueError(f"must be a {kind.__name__} (got {value!r})")
            return value
        return _Rule(check, f"a {kind.__name__}")
    return _Rule(lambda value: value, "any Python literal")


def param_hint(name):
    """Short description of the values *name* accepts, for prompts."""
    return _rule_for(name, repository_defaults().get(name)).hint


def validate_value(name, value):
    """Return *value* normalized for *name*, or raise ``ValueError`` explaining
    why it is not acceptable (e.g. ``'png'`` becomes ``'PNG'``)."""
    return _rule_for(name, repository_defaults().get(name)).check(value)


def parse_value(name, text):
    """Parse user-typed *text* for *name* and validate it.

    Python literals are accepted as written (``'PNG'``, ``[1, 2]``, ``True``).
    As a convenience, text that is not a literal is taken as a plain string, so
    ``average`` works for ``'average'``, ``true`` for ``True``, and ``A, B`` for a
    list. Raises ``ValueError`` with a message suitable for showing to the user.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("no value entered")
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        value = text
    return validate_value(name, value)


# ── Writing ──────────────────────────────────────────────────────────────────

def format_value(name, value):
    """Python source for *value*, laid out the way config.py lays it out."""
    if name == "PREFER_3D" and isinstance(value, dict):
        return format_prefer3d_dict(value)
    if name == "DICOM_HEADER_FIELDS" and isinstance(value, (list, tuple)):
        return "[\n" + "".join(f"    ({d!r}, {a!r}),\n" for d, a in value) + "]"
    return repr(value)


def assignment_lines(values, with_descriptions=False):
    """``NAME = value`` source for each item of *values* (a dict), optionally with
    each setting's description as comment lines above it."""
    descriptions = param_descriptions() if with_descriptions else {}
    chunks = []
    for name, value in values.items():
        chunk = ""
        if descriptions.get(name):
            chunk += "".join(f"# {line}".rstrip() + "\n"
                             for line in descriptions[name].splitlines())
        chunk += f"{name} = {format_value(name, value)}\n"
        chunks.append(chunk)
    return "\n".join(chunks) if with_descriptions else "".join(chunks)


def add_params_to_config(config_path, values):
    """Append *values* (``{NAME: value}``) to *config_path*.

    The settings are appended as one commented block, so nothing already in the
    file changes. The result is checked (it must parse, and each appended value
    must read back unchanged) before the file is replaced atomically; on any
    failure the original file is left untouched and the error is raised.
    """
    if not values:
        return
    target = os.path.realpath(config_path)  # write through a symlink, not over it
    with open(target, "r", encoding="utf-8") as fh:
        source = fh.read()

    banner = "# " + "=" * 76
    block = (f"{banner}\n"
             f"# Added by MRID on {datetime.date.today().isoformat()}: settings introduced\n"
             f"# after this config was created.\n"
             f"{banner}\n"
             + assignment_lines(values, with_descriptions=True))
    new_source = (source.rstrip("\n") + "\n\n\n" + block) if source.strip() else block

    compile(new_source, target, "exec")  # the whole file must still parse
    written = {}
    for node in ast.parse(block).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            written[node.targets[0].id] = ast.literal_eval(node.value)
    for name, value in values.items():
        if name not in written or written[name] != value:
            raise ValueError(f"{name} did not round-trip through the config file")

    fd, tmp = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(new_source)
            fh.flush()
            os.fsync(fh.fileno())
        shutil.copymode(target, tmp)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    logger.info(f"Added missing settings to {config_path}: {', '.join(values)}")


# ── Command-line handling ────────────────────────────────────────────────────

def manual_instructions(config_path, missing, defaults=None):
    """Text telling the user how to add *missing* settings by hand: the lines to
    paste, pre-filled with the defaults."""
    defaults = project_defaults(config_path) if defaults is None else defaults
    lines = assignment_lines({n: defaults[n] for n in missing})
    return (f"Missing setting(s): {', '.join(missing)}\n\n"
            f"To add them yourself, copy these lines into {config_path}\n"
            f"and change the values if needed:\n\n"
            f"{lines}")


def _ask(prompt, input_fn, out):
    """Print *prompt* to *out* and read a line; ``None`` on EOF or Ctrl-C."""
    out.write(prompt)
    out.flush()
    try:
        return input_fn()
    except (EOFError, KeyboardInterrupt):
        out.write("\n")
        return None


def _indent(text, prefix="      "):
    return "".join(prefix + line + "\n" for line in text.splitlines())


def prompt_for_missing_params(config_path, missing, *, input_fn=input, out=None):
    """Ask on the terminal how to fill each missing setting.

    For each one the user can accept the default, enter a value (validated, and
    asked again until valid), accept the defaults for everything remaining, or
    cancel. Returns ``{NAME: value}`` once every setting has a value and the user
    confirms, or ``None`` if they cancel (nothing is written by this function).
    """
    out = sys.stderr if out is None else out
    defaults = project_defaults(config_path)
    descriptions = param_descriptions()
    chosen = {}

    out.write(f"\nThe config file\n  {config_path}\nis missing {len(missing)} setting(s) "
              f"added in newer versions of MRID.\n")

    for index, name in enumerate(missing):
        if name in chosen:
            continue
        remaining = len(missing) - index
        default_src = format_value(name, defaults[name])
        out.write(f"\n[{index + 1}/{len(missing)}] {name}\n")
        if descriptions.get(name):
            out.write(_indent(descriptions[name]))
        out.write(f"      Valid values: {param_hint(name)}\n")
        out.write("      Default: " + default_src.replace("\n", "\n      ") + "\n")

        options = "[A]ccept default, [E]nter a value"
        keys = {"": "a", "a": "a", "e": "e", "c": "c"}
        if remaining > 1:
            options += f", accept defaults for all [R]emaining ({remaining})"
            keys["r"] = "r"
        options += ", [C]ancel"

        while True:
            answer = _ask(f"  {options} (default A): ", input_fn, out)
            if answer is None:
                answer = "c"
            choice = keys.get(answer.strip().lower())
            if choice is None:
                out.write("  Please answer " + "/".join(k.upper() for k in keys if k) + ".\n")
                continue
            if choice == "c":
                return None
            if choice == "a":
                chosen[name] = defaults[name]
                break
            if choice == "r":
                for rest in missing[index:]:
                    chosen[rest] = defaults[rest]
                break
            # choice == "e": ask for a value until it validates; blank goes back
            value = _read_value(name, input_fn, out)
            if value is _BACK:
                continue
            if value is None:
                return None
            chosen[name] = value
            break
        if name == "TARGET_SERIES":
            # A PREFER_3D still to come should suggest the labels just chosen.
            _adapt_prefer_3d(defaults, chosen["TARGET_SERIES"])
            if "PREFER_3D" in chosen and "PREFER_3D" in missing[index + 1:]:
                chosen["PREFER_3D"] = defaults["PREFER_3D"]  # accepted via [R]emaining
        if len(chosen) == len(missing):
            break

    ordered = {n: chosen[n] for n in missing}
    out.write(f"\nThese lines will be added to {config_path}:\n\n")
    out.write(_indent(assignment_lines(ordered), prefix="  "))
    answer = _ask("\nWrite them to the config file? [Y/n]: ", input_fn, out)
    if answer is None or answer.strip().lower() not in ("", "y", "yes"):
        return None
    return ordered


_BACK = object()


def _read_value(name, input_fn, out):
    """Read a value for *name* until it validates. Returns the value, ``_BACK``
    on a blank line, or ``None`` on EOF / Ctrl-C."""
    while True:
        text = _ask(f"  New value for {name} (blank to go back): ", input_fn, out)
        if text is None:
            return None
        if not text.strip():
            return _BACK
        try:
            return parse_value(name, text)
        except ValueError as e:
            out.write(f"  Invalid value: {name} {e}\n")


def resolve_missing_params_cli(config_path, missing, *, interactive, accept_defaults=False,
                               input_fn=input, out=None):
    """Fill the *missing* settings of *config_path* for a command-line run.

    * ``accept_defaults``: add the defaults without asking.
    * ``interactive``: ask on the terminal (see :func:`prompt_for_missing_params`).
    * neither: nothing can be asked, so print how to fix the file and stop.

    Chosen values are written to the config file. Returns ``{NAME: value}`` for
    the run to use, or ``None`` if the run should stop. If writing fails, the
    user is asked (interactive) whether to continue with the values for this run
    only; with ``accept_defaults`` the run continues with a warning.
    """
    out = sys.stderr if out is None else out
    defaults = project_defaults(config_path)

    if accept_defaults:
        values = {n: defaults[n] for n in missing}
        out.write(f"Adding default value(s) for {', '.join(missing)} to {config_path}\n")
    elif interactive:
        values = prompt_for_missing_params(config_path, missing, input_fn=input_fn, out=out)
        if values is None:
            out.write("\nCancelled. The config file was not changed.\n\n")
            out.write(manual_instructions(config_path, missing, defaults))
            return None
    else:
        out.write(f"\nThe config file {config_path} is missing {len(missing)} setting(s) "
                  f"added in newer versions of MRID, and this run cannot ask for them "
                  f"(input is not a terminal).\n\n")
        out.write(manual_instructions(config_path, missing, defaults))
        out.write("\nOr re-run with --accept-config-defaults to add the defaults "
                  "automatically.\n")
        return None

    try:
        add_params_to_config(config_path, values)
    except Exception as e:
        logger.error(f"Could not write the missing settings to {config_path}: {e}")
        out.write(f"\nCould not write to {config_path}:\n  {e}\n\n")
        out.write(manual_instructions(config_path, missing, values))
        if accept_defaults:
            out.write("\nContinuing this run with the default values.\n")
            return values
        if interactive:
            answer = _ask("\nContinue this run with these values without saving them? [y/N]: ",
                          input_fn, out)
            if answer is not None and answer.strip().lower() in ("y", "yes"):
                return values
        return None

    out.write(f"Updated {config_path}\n")
    return values
