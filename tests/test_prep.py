import pandas as pd
from mrid_app.prep import _preprocess_df
from mrid_app.field_detection import extract_and_clean_array_features


def test_extract_array_features():
    """Test array feature extraction using the new field_detection module."""
    df = pd.DataFrame({
        'PixelSpacing (mm)': ['1\\ 2', '[0.5, 0.5]', 'None'],
        'ImagePositionPatient (mm)': ['(1,2,3)', '1 2 3', None],
        'SeriesDescription': ['T1', 'T2', 'FLAIR']
    })

    # Create field types info for arrays
    array_fields_info = {
        'PixelSpacing (mm)': {'type': 'array', 'size': 2},
        'ImagePositionPatient (mm)': {'type': 'array', 'size': 3}
    }

    out = extract_and_clean_array_features(df.copy(), array_fields_info)

    # Expect new columns for PixelSpacing_0, PixelSpacing_1, ImagePositionPatient_0..2
    assert any(col.startswith('PixelSpacing (mm)_') for col in out.columns)
    assert any(col.startswith('ImagePositionPatient (mm)_') for col in out.columns)


def test_preprocess_df_basic():
    df = pd.DataFrame({
        'SeriesDescription': ['A', 'B'],
        'TR (Repetition Time)': [1000, 'None'],
        'TE (Echo Time)': [30, 40],
        'TI (Inversion Time)': [None, 150],
        'ScanningSequence': ['SEQ1', None],
        'SequenceVariant': ['VAR', 'VAR'],
        'ScanOptions': ['OPT', 'OPT'],
        'SequenceName': ['NAME', 'NAME'],
        'AcquisitionType': ['TYPE', 'TYPE'],
        'ImageType': ['ORIGINAL', 'ORIGINAL'],
        'Rows': [256, 256],
        'Columns': [256, 256],
        'PixelSpacing (mm)': ['1 1', '1 1'],
        'SliceThickness (mm)': [5, 5],
        'ImagePositionPatient (mm)': ['(0,0,0)', '(0,0,1)'],
        'ImageOrientationPatient': ['(1,0,0,0,1,0)', '(1,0,0,0,1,0)'],
        'SliceLocation (mm)': [0, 1],
        'SpacingBetweenSlices (mm)': ['None', '1'] ,
        'ContrastBolusAgent': [None, 'GAD']
    })
    # _preprocess_df now returns 4 values when is_training=True: (X, cat_cols, text_cols, field_types)
    X, cat_cols, text_cols, field_types = _preprocess_df(df, is_training=True)
    assert 'TR (Repetition Time)' in X.columns or any('TR (Repetition Time)' in str(col) for col in X.columns)
    assert isinstance(cat_cols, list)
    assert isinstance(text_cols, list)
    assert isinstance(field_types, dict)

