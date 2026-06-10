"""Tests for config_io: comment-preserving config.py rewriting."""

from mrid_app.config_io import (format_prefer3d_dict, replace_single_assignment,
                                rewrite_config_assignments)

SAMPLE = """# Header comment
FOO = 1  # trailing comment
BAR = [
    "a",
    "b",
]
# Footer comment
BAZ = "x"
"""


def test_replace_simple_assignment_preserves_comments():
    out = replace_single_assignment(SAMPLE, "FOO", "42")
    assert "FOO = 42  # trailing comment" in out
    assert "# Header comment" in out
    assert "# Footer comment" in out


def test_replace_multiline_assignment():
    out = replace_single_assignment(SAMPLE, "BAR", "['c']")
    assert "BAR = ['c']" in out
    assert '"a"' not in out
    assert 'BAZ = "x"' in out


def test_missing_variable_is_appended():
    out = replace_single_assignment(SAMPLE, "NEW_VAR", "7")
    assert out.rstrip().endswith("NEW_VAR = 7")


def test_rewrite_multiple_assignments():
    out = rewrite_config_assignments(SAMPLE, {"FOO": "2", "BAZ": "'y'"})
    assert "FOO = 2" in out
    assert "BAZ = 'y'" in out


def test_rewritten_source_is_valid_python():
    out = rewrite_config_assignments(SAMPLE, {"FOO": "2", "BAR": "{'k': True}"})
    namespace = {}
    exec(compile(out, "<test>", "exec"), namespace)
    assert namespace["FOO"] == 2
    assert namespace["BAR"] == {"k": True}


def test_format_prefer3d_dict():
    rendered = format_prefer3d_dict({"T1": True, "Other": False})
    parsed = eval(rendered)
    assert parsed == {"T1": True, "Other": False}
