"""
Unit tests for configuration settings and validation.
"""

def test_target_series_defined():
    """Test that TARGET_SERIES is properly defined."""
    from mrid_app.config import TARGET_SERIES
    assert TARGET_SERIES is not None
    assert isinstance(TARGET_SERIES, list)
    assert len(TARGET_SERIES) > 0
    assert "Other" in TARGET_SERIES

def test_prefer_3d_defined():
    """Test that PREFER_3D is properly defined."""
    from mrid_app.config import PREFER_3D, TARGET_SERIES
    assert PREFER_3D is not None
    assert isinstance(PREFER_3D, dict)

    # Check that all target series have a preference
    for series in TARGET_SERIES:
        assert series in PREFER_3D, f"{series} not in PREFER_3D"
        assert isinstance(PREFER_3D[series], bool)

def test_override_threshold_valid():
    """Test that OVERRIDE_THRESHOLD is valid."""
    from mrid_app.config import OVERRIDE_THRESHOLD
    assert OVERRIDE_THRESHOLD is not None
    assert isinstance(OVERRIDE_THRESHOLD, (int, float))
    assert 0.0 <= OVERRIDE_THRESHOLD <= 1.0

def test_dicom_header_fields_format():
    """Test that DICOM_HEADER_FIELDS is properly formatted."""
    from mrid_app.config import DICOM_HEADER_FIELDS
    assert DICOM_HEADER_FIELDS is not None
    assert isinstance(DICOM_HEADER_FIELDS, list)
    assert len(DICOM_HEADER_FIELDS) > 0

    # Check format of each field
    for field in DICOM_HEADER_FIELDS:
        assert isinstance(field, tuple)
        assert len(field) == 2
        display_name, dicom_attr = field
        assert isinstance(display_name, str)
        assert isinstance(dicom_attr, str)
        assert len(display_name) > 0
        assert len(dicom_attr) > 0

def test_cnn_hyperparameters():
    """Test that CNN hyperparameters are valid."""
    from mrid_app.config import IMAGE_SIZE, NUM_EPOCHS, BATCH_SIZE, LR

    assert IMAGE_SIZE > 0
    assert isinstance(IMAGE_SIZE, int)

    assert NUM_EPOCHS > 0
    assert isinstance(NUM_EPOCHS, int)

    assert BATCH_SIZE > 0
    assert isinstance(BATCH_SIZE, int)

    assert LR > 0
    assert isinstance(LR, (int, float))

def test_rf_hyperparameters():
    """Test that Random Forest hyperparameters are valid."""
    from mrid_app.config import N_ESTIMATORS
    assert N_ESTIMATORS > 0
    assert isinstance(N_ESTIMATORS, int)

def test_field_detection_thresholds():
    """Test that field detection thresholds are valid."""
    from mrid_app.config import CATEGORICAL_UNIQUE_THRESHOLD, TEXT_MIN_WORDS, ARRAY_SIZE_THRESHOLD

    assert CATEGORICAL_UNIQUE_THRESHOLD > 0
    assert isinstance(CATEGORICAL_UNIQUE_THRESHOLD, int)

    assert TEXT_MIN_WORDS >= 1
    assert isinstance(TEXT_MIN_WORDS, int)

    assert ARRAY_SIZE_THRESHOLD >= 1
    assert isinstance(ARRAY_SIZE_THRESHOLD, int)

def test_mr_acquisition_type_in_header_fields():
    """Test that MRAcquisitionType is in DICOM_HEADER_FIELDS (needed for PREFER_3D)."""
    from mrid_app.config import DICOM_HEADER_FIELDS

    # Check if MRAcquisitionType is in the fields
    field_attrs = [attr for _, attr in DICOM_HEADER_FIELDS]
    assert "MRAcquisitionType" in field_attrs, "MRAcquisitionType needed for PREFER_3D feature"

