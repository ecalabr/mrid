"""
Unit tests for automatic DICOM header field detection.
"""
import pandas as pd
import numpy as np

def test_detect_field_type_numerical():
    """Test detection of numerical fields."""
    from mrid_app.field_detection import detect_field_type

    # Create numerical data
    series = pd.Series([100, 200, 300, 400, 500])
    result = detect_field_type(series, "TestField")
    assert result == 'numerical'

def test_detect_field_type_array():
    """Test detection of array fields."""
    from mrid_app.field_detection import detect_field_type

    # Create array data
    series = pd.Series(['(1.0, 2.0)', '(1.5, 2.5)', '(2.0, 3.0)'])
    result = detect_field_type(series, "TestArray")
    assert result == 'array'

def test_detect_field_type_categorical():
    """Test detection of categorical fields."""
    from mrid_app.field_detection import detect_field_type

    # Create categorical data
    series = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'])
    result = detect_field_type(series, "TestCategorical")
    assert result in ['categorical', 'text']  # Could be either depending on thresholds

def test_detect_array_size():
    """Test detection of array size."""
    from mrid_app.field_detection import detect_array_size

    # Create array data with consistent size
    series = pd.Series(['(1.0, 2.0)', '(1.5, 2.5)', '(2.0, 3.0)'])
    result = detect_array_size(series, "TestArray")
    assert result == 2

def test_analyze_dataframe_fields():
    """Test analysis of multiple DataFrame fields."""
    from mrid_app.field_detection import analyze_dataframe_fields

    df = pd.DataFrame({
        'Numerical': [100, 200, 300],
        'Array': ['(1, 2)', '(3, 4)', '(5, 6)'],
        'Categorical': ['A', 'B', 'A']
    })

    result = analyze_dataframe_fields(df, df.columns.tolist())

    assert isinstance(result, dict)
    assert 'Numerical' in result
    assert 'Array' in result
    assert 'Categorical' in result

    # Check types
    assert result['Numerical']['type'] == 'numerical'
    assert result['Array']['type'] == 'array'
    assert result['Array']['size'] == 2

def test_extract_and_clean_array_features():
    """Test extraction and cleaning of array features."""
    from mrid_app.field_detection import extract_and_clean_array_features

    df = pd.DataFrame({
        'TestArray': ['(1.0, 2.0)', '(3.0, 4.0)', '(5.0, 6.0)'],
        'Other': ['A', 'B', 'C']
    })

    array_info = {
        'TestArray': {'type': 'array', 'size': 2}
    }

    result = extract_and_clean_array_features(df, array_info)

    # Should have expanded columns
    assert 'TestArray_0' in result.columns
    assert 'TestArray_1' in result.columns
    assert 'Other' in result.columns

    # Original array column should be removed
    assert 'TestArray' not in result.columns

def test_preprocess_by_type():
    """Test preprocessing by detected types."""
    from mrid_app.field_detection import preprocess_by_type, analyze_dataframe_fields

    df = pd.DataFrame({
        'Numerical': [100, 200, 300],
        'Text': ['description one', 'description two', 'description three'],
        'Categorical': ['A', 'B', 'A']
    })

    # First analyze
    field_types = analyze_dataframe_fields(df, df.columns.tolist())

    # Then preprocess
    result_df, cat_cols, text_cols = preprocess_by_type(df, field_types)

    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(cat_cols, list)
    assert isinstance(text_cols, list)

def test_field_detection_handles_missing_values():
    """Test that field detection handles missing values correctly."""
    from mrid_app.field_detection import detect_field_type

    # Series with missing values
    series = pd.Series([100, None, 300, np.nan, 500, 'None'])
    result = detect_field_type(series, "TestMissing")

    # Should still be detected as numerical
    assert result == 'numerical'

def test_field_detection_empty_series():
    """Test field detection with empty series."""
    from mrid_app.field_detection import detect_field_type

    series = pd.Series([])
    result = detect_field_type(series, "TestEmpty")

    # Should return some default type
    assert result in ['numerical', 'array', 'categorical', 'text']

