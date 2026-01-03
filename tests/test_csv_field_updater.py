"""
Unit tests for CSV field updater functionality.
"""

def test_get_expected_field_names():
    """Test that get_expected_field_names returns correct field names."""
    from mrid_app.csv_field_updater import get_expected_field_names
    from mrid_app.config import DICOM_HEADER_FIELDS

    field_names = get_expected_field_names()

    assert isinstance(field_names, list)
    assert len(field_names) == len(DICOM_HEADER_FIELDS)

    # Should match display names from config
    expected = [field[0] for field in DICOM_HEADER_FIELDS]
    assert field_names == expected

def test_check_missing_fields_nonexistent():
    """Test check_missing_fields with non-existent file."""
    from mrid_app.csv_field_updater import check_missing_fields

    result = check_missing_fields("/nonexistent/file.csv")
    assert isinstance(result, list)

def test_scan_for_missing_fields_invalid_dir():
    """Test scan_for_missing_fields with invalid directory."""
    from mrid_app.csv_field_updater import scan_for_missing_fields

    result = scan_for_missing_fields("/nonexistent/directory")
    assert isinstance(result, dict)
    assert len(result) == 0

def test_extract_dicom_field_value_invalid():
    """Test extract_dicom_field_value with invalid path."""
    from mrid_app.csv_field_updater import extract_dicom_field_value

    result = extract_dicom_field_value("/nonexistent/file.dcm", "SeriesDescription")
    assert result == "None"

def test_safe_get_dicom_attribute_exists():
    """Test that safe_get_dicom_attribute function exists."""
    from mrid_app.csv_field_updater import safe_get_dicom_attribute
    assert callable(safe_get_dicom_attribute)

