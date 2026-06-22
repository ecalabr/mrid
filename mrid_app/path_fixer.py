"""
Repair stored absolute paths after a project's input and/or output directory is
moved or renamed.

MRID stores absolute paths in three places under the output directory:
  - ``hierarchy.yaml`` — per-series ``dicom_path`` (a symlink under
    ``dicom_symlinks/`` or an original input DICOM) and ``thumbnail_path``
    (under ``series_thumbnails/``).
  - ``dicom_symlinks/`` — symlinks whose *targets* point at the original input
    DICOM files.
  - ``classified_outputs/<study>/<study>_classification-results.csv`` — a
    ``Path`` column referencing the series' DICOM (symlink or input).

When the output directory moves, paths that live *inside* it (symlinks,
thumbnails, the symlink-based ``dicom_path``/``Path``) are rebased onto the new
output directory deterministically (they keep their known relative location).
When the input directory moves, symlink targets and any direct input paths are
re-discovered under the new input directory by matching the longest existing
path suffix. Anything that still cannot be found is reported as *missing* so the
caller can offer to locate it (copying from a user-provided source) or drop it.
"""

import os
import csv
import shutil

import yaml

from .logging_config import get_logger
from .utils import get_classified_outputs_dir, get_study_csv_name, extract_study_id_from_path

logger = get_logger()

# Subdirectories that live under the output directory. A stored path containing
# one of these can be rebased onto a new output directory by keeping everything
# from the token onward.
_OUTPUT_SUBDIRS = ("classified_outputs", "series_thumbnails", "dicom_symlinks")


def rebase_output_path(path, new_output_dir):
    """If *path* lives under a known output subdirectory, rebase it onto
    *new_output_dir* (keeping the subdir-relative tail). Otherwise return it
    unchanged."""
    if not path:
        return path
    norm = path.replace("\\", "/")
    for token in _OUTPUT_SUBDIRS:
        marker = "/" + token + "/"
        i = norm.find(marker)
        if i != -1:
            tail = norm[i + 1:]                  # e.g. "dicom_symlinks/AB/x.dcm"
            return os.path.normpath(os.path.join(new_output_dir, tail))
    return path


def find_under_root(target, root):
    """Locate *target*'s file under *root* by matching the longest path suffix
    that exists. Returns the found path, or None.

    e.g. target ``/old/data/sub/f.dcm`` with root ``/new/data`` tries
    ``/new/data/old/data/sub/f.dcm`` ... ``/new/data/sub/f.dcm`` ...
    ``/new/data/f.dcm`` and returns the first (longest-tail) match."""
    if not target or not root or not os.path.isdir(root):
        return None
    parts = [p for p in os.path.normpath(target).replace("\\", "/").split("/") if p and p != ".."]
    for start in range(len(parts)):
        cand = os.path.join(root, *parts[start:])
        if os.path.exists(cand):
            return cand
    return None


def _resolve_input_path(target, new_input_dir, source_root):
    """Resolve a (possibly broken) input DICOM path. Returns (resolved_path,
    copied_bool) or (None, False) if it cannot be found.

    Order: keep it if it already exists; else find it under the new input dir;
    else (if a source root is given) find it there and copy it into the new
    input dir, preserving the matched relative structure."""
    if target and os.path.exists(target):
        return target, False
    found = find_under_root(target, new_input_dir)
    if found:
        return found, False
    if source_root:
        src = find_under_root(target, source_root)
        if src:
            # Mirror the path relative to source_root into the new input dir.
            rel = os.path.relpath(src, source_root)
            dest = os.path.join(new_input_dir, rel)
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)
                return dest, True
            except Exception as e:
                logger.warning(f"Failed to copy missing data {src} -> {dest}: {e}")
    return None, False


def _fix_symlinks(output_dir, new_input_dir, source_root, remove_missing, report):
    """Re-point symlinks in dicom_symlinks/ whose targets moved. Returns a set of
    symlink directories that remain broken (for optional removal)."""
    sym_base = os.path.join(output_dir, "dicom_symlinks")
    broken_dirs = set()
    if not os.path.isdir(sym_base):
        return broken_dirs
    for dirpath, _dirs, files in os.walk(sym_base):
        for name in files:
            link = os.path.join(dirpath, name)
            if not os.path.islink(link):
                continue
            try:
                target = os.readlink(link)
            except OSError:
                continue
            if os.path.exists(link):
                continue  # target still valid
            resolved, _copied = _resolve_input_path(target, new_input_dir, source_root)
            if resolved:
                try:
                    os.remove(link)
                    os.symlink(resolved, link)
                    report["symlinks_fixed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to relink {link}: {e}")
                    report["missing"].append(target)
                    broken_dirs.add(dirpath)
            else:
                report["missing"].append(target)
                broken_dirs.add(dirpath)
    if remove_missing:
        for d in broken_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
                report["removed_symlink_dirs"] += 1
            except Exception as e:
                logger.warning(f"Failed to remove broken symlink dir {d}: {e}")
    return broken_dirs


def _fix_hierarchy(output_dir, new_input_dir, source_root, remove_missing, report):
    """Rebase thumbnail/dicom paths in hierarchy.yaml and drop unresolved series
    when remove_missing is set. Returns the resolved set of dicom paths kept."""
    hpath = os.path.join(output_dir, "hierarchy.yaml")
    if not os.path.exists(hpath):
        return
    try:
        with open(hpath, "r") as f:
            hierarchy = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not read hierarchy.yaml: {e}")
        return

    changed = False
    for _mrn, studies in list(hierarchy.items()):
        if not isinstance(studies, dict):
            continue
        for _suid, study_data in list(studies.items()):
            series = (study_data or {}).get("series", {})
            for desc, entries in list(series.items()):
                kept = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        kept.append(entry)
                        continue
                    # Thumbnails live in the output dir — always rebase.
                    tp = entry.get("thumbnail_path")
                    if tp:
                        new_tp = rebase_output_path(tp, output_dir)
                        if new_tp != tp:
                            entry["thumbnail_path"] = new_tp
                            changed = True
                    dp = entry.get("dicom_path")
                    if dp and ("/dicom_symlinks/" in dp.replace("\\", "/")):
                        # Symlink-based: rebase to the (current) output dir; the
                        # symlink itself was re-pointed in _fix_symlinks. Keep it
                        # only if it now resolves (else it's missing).
                        new_dp = rebase_output_path(dp, output_dir)
                        if new_dp != dp:
                            entry["dicom_path"] = new_dp
                            changed = True
                        if os.path.exists(new_dp):
                            kept.append(entry)
                        else:
                            # The underlying missing input file is reported by
                            # _fix_symlinks; here we only drop the dangling entry.
                            if remove_missing:
                                report["removed_series"] += 1
                                changed = True
                            else:
                                kept.append(entry)
                    elif dp:
                        # Direct input path — re-discover under the new input.
                        resolved, _c = _resolve_input_path(dp, new_input_dir, source_root)
                        if resolved:
                            if resolved != dp:
                                entry["dicom_path"] = resolved
                                changed = True
                            kept.append(entry)
                        else:
                            report["missing"].append(dp)
                            if not remove_missing:
                                kept.append(entry)  # leave it; user may locate later
                            else:
                                report["removed_series"] += 1
                                changed = True
                    else:
                        kept.append(entry)
                if kept:
                    series[desc] = kept
                else:
                    del series[desc]
                    changed = True

    if changed:
        try:
            with open(hpath, "w") as f:
                yaml.safe_dump(hierarchy, f, indent=2, default_flow_style=False,
                               sort_keys=False, allow_unicode=True)
            report["hierarchy_updated"] = True
        except Exception as e:
            logger.warning(f"Could not write hierarchy.yaml: {e}")


def _fix_csvs(output_dir, new_input_dir, source_root, remove_missing, report):
    """Rebase/repair the Path column in each classification CSV; drop rows whose
    Path cannot be resolved when remove_missing is set."""
    classified_dir = get_classified_outputs_dir(output_dir)
    if not os.path.isdir(classified_dir):
        return
    for folder in os.listdir(classified_dir):
        folder_path = os.path.join(classified_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        study_id = extract_study_id_from_path(folder_path)
        csv_path = os.path.join(folder_path, get_study_csv_name(study_id))
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
            continue
        if not fieldnames or "Path" not in fieldnames:
            continue

        new_rows = []
        changed = False
        for row in rows:
            p = row.get("Path", "")
            if p and ("/dicom_symlinks/" in p.replace("\\", "/")):
                np = rebase_output_path(p, output_dir)
                if np != p:
                    row["Path"] = np
                    changed = True
                if os.path.exists(np):
                    new_rows.append(row)
                else:
                    # Missing input is reported via _fix_symlinks; drop the
                    # dangling row only when removing.
                    if remove_missing:
                        report["removed_rows"] += 1
                        changed = True
                    else:
                        new_rows.append(row)
            elif p:
                resolved, _c = _resolve_input_path(p, new_input_dir, source_root)
                if resolved:
                    if resolved != p:
                        row["Path"] = resolved
                        changed = True
                    new_rows.append(row)
                else:
                    report["missing"].append(p)
                    if remove_missing:
                        report["removed_rows"] += 1
                        changed = True
                    else:
                        new_rows.append(row)
            else:
                new_rows.append(row)

        if changed:
            try:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(new_rows)
                report["csvs_updated"] += 1
            except Exception as e:
                logger.warning(f"Could not write {csv_path}: {e}")


def fix_project_paths(output_dir, new_input_dir, source_root=None, remove_missing=False):
    """Repair stored paths after the input/output directory moved.

    Args:
        output_dir: the (current/new) output directory.
        new_input_dir: the (current/new) input directory where DICOMs now live.
        source_root: optional folder to copy still-missing DICOMs from (the
            "highest level missing path"); files are auto-discovered under it
            and copied into new_input_dir.
        remove_missing: if True, drop series/rows/symlinks that cannot be
            resolved instead of leaving them.

    Returns a report dict summarizing what was fixed and what is still missing.
    """
    report = {
        "symlinks_fixed": 0,
        "removed_symlink_dirs": 0,
        "hierarchy_updated": False,
        "csvs_updated": 0,
        "removed_series": 0,
        "removed_rows": 0,
        "missing": [],
    }
    if not os.path.isdir(output_dir):
        report["error"] = f"Output directory does not exist: {output_dir}"
        return report

    _fix_symlinks(output_dir, new_input_dir, source_root, remove_missing, report)
    _fix_hierarchy(output_dir, new_input_dir, source_root, remove_missing, report)
    _fix_csvs(output_dir, new_input_dir, source_root, remove_missing, report)

    # De-duplicate the missing list (the same source can be referenced widely).
    report["missing"] = sorted(set(report["missing"]))
    report["missing_count"] = len(report["missing"])
    return report
