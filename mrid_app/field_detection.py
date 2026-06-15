"""
Automatic DICOM header field type detection and encoding.

This module analyzes DICOM header fields and automatically determines
whether they should be treated as:
- Numerical features (single numbers)
- Numerical arrays (lists of numbers)
- Categorical variables (limited discrete values)
- Text features (free-form text for semantic encoding)
"""

import re
import numpy as np
import pandas as pd
from .logging_config import get_logger

logger = get_logger()


def detect_field_type(series, field_name):
    """
    Automatically detect the type of a DICOM header field.

    Args:
        series: pandas Series containing the field values
        field_name: Name of the field (for logging)

    Returns:
        str: One of 'numerical', 'array', 'categorical', 'text'
    """
    from .config import CATEGORICAL_UNIQUE_THRESHOLD, TEXT_MIN_WORDS, ARRAY_SIZE_THRESHOLD

    # Remove None/NaN values for analysis
    valid_values = series.dropna()
    valid_values = valid_values[valid_values != 'None']
    valid_values = valid_values[valid_values != '']

    if len(valid_values) == 0:
        # Default to numerical if no valid values
        return 'numerical'

    # Check if values contain multiple numbers (array)
    sample_values = valid_values.head(min(100, len(valid_values)))
    array_count = 0
    for val in sample_values:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
        if len(numbers) > ARRAY_SIZE_THRESHOLD:
            array_count += 1

    if array_count / len(sample_values) > 0.5:
        return 'array'

    # Try to convert to numeric
    try:
        numeric_series = pd.to_numeric(valid_values, errors='coerce')
        numeric_count = numeric_series.notna().sum()

        # If most values can be converted to numbers, it's numerical
        if numeric_count / len(valid_values) > 0.8:
            return 'numerical'
    except Exception:
        pass

    # Check unique value count for categorical detection
    unique_count = valid_values.nunique()

    if unique_count <= CATEGORICAL_UNIQUE_THRESHOLD:
        # Check if it looks like text (multiple words)
        word_counts = []
        for val in sample_values:
            words = str(val).split()
            word_counts.append(len(words))

        avg_words = np.mean(word_counts) if word_counts else 0

        if avg_words >= TEXT_MIN_WORDS:
            return 'text'
        else:
            return 'categorical'
    else:
        # Many unique values, likely text
        return 'text'


def detect_array_size(series, field_name):
    """
    Detect the size of array values in a field.

    Args:
        series: pandas Series containing array values
        field_name: Name of the field

    Returns:
        int: Expected array size
    """
    valid_values = series.dropna()
    valid_values = valid_values[valid_values != 'None']

    sizes = []
    for val in valid_values.head(100):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
        sizes.append(len(numbers))

    if not sizes:
        return 2  # Default size

    # Return the most common size
    from collections import Counter
    most_common_size = Counter(sizes).most_common(1)[0][0]
    return max(1, most_common_size)


def analyze_dataframe_fields(df, field_names):
    """
    Analyze a DataFrame and determine the type of each field.

    Args:
        df: pandas DataFrame with DICOM header fields
        field_names: List of field names to analyze

    Returns:
        dict: Mapping of field_name -> {'type': str, 'size': int (for arrays)}
    """
    field_types = {}

    for field_name in field_names:
        if field_name not in df.columns:
            continue

        field_type = detect_field_type(df[field_name], field_name)

        field_info = {'type': field_type}

        if field_type == 'array':
            array_size = detect_array_size(df[field_name], field_name)
            field_info['size'] = array_size

        field_types[field_name] = field_info
        logger.debug(f"  {field_name}: {field_type}" + (f" (size={field_info.get('size')})" if field_type == 'array' else ""))

    return field_types


def extract_and_clean_array_features(df, array_fields_info):
    """
    Extract and expand array fields into individual columns.

    Args:
        df: pandas DataFrame
        array_fields_info: dict mapping field_name -> {'type': 'array', 'size': int}

    Returns:
        pandas DataFrame with arrays expanded into columns
    """
    for field_name, info in array_fields_info.items():
        if info['type'] != 'array' or field_name not in df.columns:
            continue

        size = info['size']

        def parse_array_string(val):
            if pd.isna(val) or val is None or val == 'None' or val == '':
                return [np.nan] * size
            try:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
                float_list = [float(n) for n in numbers[:size]]
                float_list.extend([np.nan] * (size - len(float_list)))
                return float_list[:size]
            except Exception:
                return [np.nan] * size

        parsed_data = df[field_name].apply(parse_array_string).tolist()
        parsed_df = pd.DataFrame(
            parsed_data,
            index=df.index,
            columns=[f"{field_name}_{i}" for i in range(size)]
        )

        df = df.drop(columns=[field_name])
        df = pd.concat([df, parsed_df], axis=1)

    return df


def preprocess_by_type(df, field_types):
    """
    Preprocess DataFrame fields based on their detected types.

    Args:
        df: pandas DataFrame
        field_types: dict from analyze_dataframe_fields

    Returns:
        tuple: (preprocessed_df, categorical_cols, text_cols)
    """
    # Separate fields by type
    numerical_fields = []
    array_fields = {}
    categorical_fields = []
    text_fields = []

    for field_name, info in field_types.items():
        if field_name not in df.columns:
            continue

        field_type = info['type']
        if field_type == 'numerical':
            numerical_fields.append(field_name)
        elif field_type == 'array':
            array_fields[field_name] = info
        elif field_type == 'categorical':
            categorical_fields.append(field_name)
        elif field_type == 'text':
            text_fields.append(field_name)

    # Extract and expand array fields
    df_processed = extract_and_clean_array_features(df.copy(), array_fields)

    # Get all column names after array expansion
    all_cols = df_processed.columns.tolist()

    # Array fields have been expanded, so add their column names to numerical
    for field_name, info in array_fields.items():
        size = info['size']
        expanded_cols = [f"{field_name}_{i}" for i in range(size)]
        numerical_fields.extend([col for col in expanded_cols if col in all_cols])

    # Process numerical fields
    for col in numerical_fields:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(-1.0).astype(float)

    # Process categorical fields
    for col in categorical_fields:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).fillna('MISSING').str.strip()

    # Process text fields
    for col in text_fields:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('MISSING').astype(str).str.replace('_', ' ').str.strip()

    return df_processed, categorical_fields, text_fields

