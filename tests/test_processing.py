"""
Unit tests for processing module functions.
"""
import numpy as np

def test_check_is_3d_acquisition_exists():
    """Test that check_is_3d_acquisition function exists."""
    from mrid_app.processing import check_is_3d_acquisition
    assert callable(check_is_3d_acquisition)

def test_check_is_3d_acquisition_invalid_path():
    """Test check_is_3d_acquisition with invalid path."""
    from mrid_app.processing import check_is_3d_acquisition

    # Should return False for non-existent path
    result = check_is_3d_acquisition("/nonexistent/path/to/file.dcm")
    assert result is False
    assert isinstance(result, bool)

def test_optimize_study_selections_exists():
    """Test that optimize_study_selections function exists."""
    from mrid_app.processing import optimize_study_selections
    assert callable(optimize_study_selections)

def test_optimize_study_selections_no_encoder():
    """Test optimize_study_selections with no encoder (fallback case)."""
    from mrid_app.processing import optimize_study_selections

    study_dict = {
        'Series1': ['/path/to/file1.dcm'],
        'Series2': ['/path/to/file2.dcm']
    }

    result, predictions = optimize_study_selections(
        study_dict, "DICOM Header Model",
        None, None, None, None, None
    )

    # Should return assignments and predictions
    assert isinstance(result, dict)
    assert isinstance(predictions, dict)

    # All series should be assigned to 'Other' when no encoder
    assert len(result) == 2
    for path, label in result.items():
        assert label == "Other"

def test_convert_series_to_nifti_exists():
    """Test that convert_series_to_nifti function exists."""
    from mrid_app.processing import convert_series_to_nifti
    assert callable(convert_series_to_nifti)

def test_load_center_slice_exists():
    """Test that load_center_slice function exists."""
    from mrid_app.processing import load_center_slice
    assert callable(load_center_slice)

def test_load_center_slice_invalid_path():
    """Test load_center_slice with invalid path."""
    from mrid_app.processing import load_center_slice

    result = load_center_slice(["/nonexistent/path/to/file.dcm"])
    assert result is None

def test_full_auto_classify_study_exists():
    """Test that full_auto_classify_study function exists."""
    from mrid_app.processing import full_auto_classify_study
    assert callable(full_auto_classify_study)

