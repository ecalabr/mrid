"""
Tests for path_fixer's handling of newline-corrupted stored paths.

Older scans could bake a stray CR/LF (from an unsanitized PatientID) into symlink
directory and thumbnail file names, and therefore into the paths stored in
hierarchy.yaml and the classification CSVs. "Fix broken filepaths" must clean
those names on disk and rewrite the stored paths so they resolve again.
"""
import os
import csv
import tempfile
import shutil

import yaml


def test_rebase_output_path_collapses_newlines():
    from mrid_app.path_fixer import rebase_output_path
    p = "/old/mrid_out/dicom_symlinks/AB\rCD-uid/IM-0001-0001.dcm"
    out = rebase_output_path(p, "/new/mrid_out")
    assert "\r" not in out and "\n" not in out
    assert out == "/new/mrid_out/dicom_symlinks/AB-CD-uid/IM-0001-0001.dcm"


def test_fix_project_paths_sanitizes_newline_names():
    from mrid_app import path_fixer

    old = "/old/mrid_out"
    cr_dir = "P1\r99-none-1.3.7"          # symlink dir name with an embedded CR
    clean_dir = "P1-99-none-1.3.7"
    cr_thumb = "P1\r99_a_b.jpg"
    clean_thumb = "P1-99_a_b.jpg"
    fname = "IM-0001-0001.dcm"

    repro = tempfile.mkdtemp()
    try:
        out = os.path.join(repro, "mrid_out")
        inp = os.path.join(repro, "in")

        # on-disk symlink dir (CR in name) with a broken symlink -> old input
        symdir = os.path.join(out, "dicom_symlinks", cr_dir)
        os.makedirs(symdir)
        os.symlink(os.path.join(old, "raw", cr_dir, fname), os.path.join(symdir, fname))
        # thumbnail file with CR in name
        os.makedirs(os.path.join(out, "series_thumbnails"))
        open(os.path.join(out, "series_thumbnails", cr_thumb), "w").write("jpg")
        # moved input so the symlink can be re-pointed
        real_input = os.path.join(inp, cr_dir, fname)
        os.makedirs(os.path.dirname(real_input))
        open(real_input, "w").write("dcm")

        # CSV + hierarchy referencing the old CR paths
        study = "S1"
        sdir = os.path.join(out, "classified_outputs", study)
        os.makedirs(sdir)
        csvp = os.path.join(sdir, f"{study}_classification-results.csv")
        with open(csvp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Classification", "Path", "SeriesDescription"])
            w.writerow(["FLAIR", f"{old}/dicom_symlinks/{cr_dir}/{fname}", "ax flair"])
        hier = {"P1\r99": {"1.3": {"accn": "none", "series": {"ax flair": [
            {"dicom_path": f"{old}/dicom_symlinks/{cr_dir}/{fname}",
             "thumbnail_path": f"{old}/series_thumbnails/{cr_thumb}"}]}}}}
        with open(os.path.join(out, "hierarchy.yaml"), "w") as f:
            yaml.safe_dump(hier, f)

        rep = path_fixer.fix_project_paths(out, inp)

        # on-disk names cleaned
        assert rep["sanitized_names"] == 2
        assert os.path.isdir(os.path.join(out, "dicom_symlinks", clean_dir))
        assert not os.path.isdir(os.path.join(out, "dicom_symlinks", cr_dir))
        assert os.path.isfile(os.path.join(out, "series_thumbnails", clean_thumb))

        # symlink re-pointed to the moved input
        assert rep["symlinks_fixed"] == 1

        # CSV Path cleaned and resolvable
        with open(csvp, newline="") as f:
            row = next(csv.DictReader(f))
        assert "\r" not in row["Path"] and "\n" not in row["Path"]
        assert os.path.exists(row["Path"])

        # hierarchy paths cleaned and resolvable
        h = yaml.safe_load(open(os.path.join(out, "hierarchy.yaml")))
        entry = list(list(h.values())[0].values())[0]["series"]["ax flair"][0]
        for key in ("dicom_path", "thumbnail_path"):
            assert "\r" not in entry[key] and "\n" not in entry[key]
            assert os.path.exists(entry[key])

        assert rep["missing_count"] == 0
    finally:
        shutil.rmtree(repro, ignore_errors=True)
