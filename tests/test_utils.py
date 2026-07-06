"""
Unit tests for utils naming helpers.
"""


def test_format_study_name_strips_newlines():
    """Newlines leaking in from DICOM header values must not end up literally in
    output names; they are replaced with '-'."""
    from mrid_app.utils import format_study_name
    name = format_study_name("A\nB", "U", "C", "{mrn}_{study_uid}_{accn}")
    assert "\n" not in name and "\r" not in name
    assert name == "A-B_U_C"


def test_format_study_name_crlf_single_dash():
    """A CR/LF run collapses to a single dash (never two or more in a row)."""
    from mrid_app.utils import format_study_name
    name = format_study_name("A\r\nB", "U", "C", "{mrn}")
    assert name == "A-B"
    assert "--" not in name


def test_format_study_name_newline_adjacent_to_template_dash():
    """A newline next to a template dash still yields a single dash."""
    from mrid_app.utils import format_study_name
    name = format_study_name("A", "U", "\nB", "{mrn}-{accn}")
    assert name == "A-B"
    assert "--" not in name


def test_format_study_name_interleaved_newlines_and_dashes():
    """Interleaved newlines and dashes collapse to one dash."""
    from mrid_app.utils import format_study_name
    name = format_study_name("A\n-\nB", "U", "C", "{mrn}")
    assert name == "A-B"


def test_format_study_name_preserves_existing_dashes():
    """Dashes that are not produced by newline replacement are left untouched."""
    from mrid_app.utils import format_study_name
    assert format_study_name("12-34", "U", "C", "{mrn}") == "12-34"
    # A pre-existing double dash with no newline is the user's own value/template.
    assert format_study_name("A--B", "U", "C", "{mrn}") == "A--B"


def test_get_symlink_dir_name_sanitizes_newlines():
    """A CR/LF leaking in from the PatientID must not end up in the symlink
    directory name (which is stored in hierarchy.yaml and the CSVs)."""
    from mrid_app.utils import get_symlink_dir_name
    name = get_symlink_dir_name("A\rB", "1.3.99", "1.3.55", accn="none")
    assert "\r" not in name and "\n" not in name and "--" not in name


def test_csv_name_is_newline_free():
    """The classification CSV name derives from the study dir name and must be
    newline-free too."""
    from mrid_app.utils import format_study_name, get_study_csv_name
    name = format_study_name("M\rN", "U\n", "A", "{mrn}_{study_uid}_{accn}")
    csv = get_study_csv_name(name)
    assert "\n" not in csv and "\r" not in csv and "--" not in csv
