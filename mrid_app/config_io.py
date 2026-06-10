"""Helpers for reading and rewriting config.py files while preserving
comments and formatting. Used by the Analyze workflow and the config wizard."""

import ast

from .logging_config import get_logger

logger = get_logger()


def format_prefer3d_dict(prefer3d_dict: dict) -> str:
    """
    Render a PREFER_3D dict as a nicely-indented Python literal, e.g.::

        {
            "T1": True,
            "T2": False,
        }
    """
    lines = ["{"]
    for k, v in prefer3d_dict.items():
        lines.append(f'    "{k}": {v},')
    lines.append("}")
    return "\n".join(lines)


def rewrite_config_assignments(source: str, replacements: dict) -> str:
    """
    Replace top-level simple assignment statements in *source* for each variable
    name in *replacements* ({var_name: new_rhs_repr_string}). Variables not
    present in the source are appended. All other content — comments, ordering,
    formatting — is preserved.
    """
    for var_name, new_rhs in replacements.items():
        source = replace_single_assignment(source, var_name, new_rhs)
    return source


def replace_single_assignment(source: str, var_name: str, new_rhs: str) -> str:
    """
    Replace the RHS of a single top-level ``var_name = <expr>`` assignment.

    If the variable is not found in the source it is *appended* at the end.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back: crude regex replacement if the file has a syntax error
        import re
        pattern = re.compile(
            r'^' + re.escape(var_name) + r'\s*=\s*.+?(?=\n[A-Z_#\n]|\Z)',
            re.MULTILINE | re.DOTALL
        )
        new_stmt = f"{var_name} = {new_rhs}"
        if pattern.search(source):
            return pattern.sub(new_stmt, source, count=1)
        return source + f"\n{new_stmt}\n"

    # Find the target Assign node at module level
    target_node = None
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == var_name:
                target_node = node
                break

    if target_node is None:
        # Variable not present — append it
        return source.rstrip() + f"\n\n{var_name} = {new_rhs}\n"

    # ast gives us line numbers (1-based). Translate to character offsets.
    lines = source.splitlines(keepends=True)

    stmt_start_line = target_node.lineno - 1        # 0-based index
    stmt_end_line = target_node.end_lineno - 1      # 0-based index (inclusive)

    # Preserve any trailing comment on the last line of the assignment.
    last_line = lines[stmt_end_line]
    end_col = target_node.end_col_offset
    trailing = last_line[end_col:].rstrip('\n')

    new_stmt = f"{var_name} = {new_rhs}"
    if trailing.lstrip().startswith('#'):
        new_stmt += "  " + trailing.lstrip()
    new_stmt += "\n"

    before = "".join(lines[:stmt_start_line])
    after = "".join(lines[stmt_end_line + 1:])
    return before + new_stmt + after


def get_default_config_path():
    """Path to the repository default config.py (the wizard's template)."""
    from . import config as config_module
    path = config_module.__file__
    if path.endswith('.pyc'):
        path = path[:-1]
    return path
