import time
import pandas as pd
from .config import SEMANTIC_MODEL_NAME, DICOM_HEADER_FIELDS
from .field_detection import analyze_dataframe_fields, preprocess_by_type
from .logging_config import get_logger

logger = get_logger()


def _get_semantic_model():
    from . import config
    if config.SEMANTIC_MODEL is None:
        try:
            import logging
            from sentence_transformers import SentenceTransformer
            # Suppress noisy key-mismatch warnings from transformers during model load
            _transformers_logger = logging.getLogger("transformers.modeling_utils")
            _prev_level = _transformers_logger.level
            _transformers_logger.setLevel(logging.ERROR)
            config.SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)
            _transformers_logger.setLevel(_prev_level)
            logger.info(f"Loaded semantic model: {SEMANTIC_MODEL_NAME}")
        except Exception:
            config.SEMANTIC_MODEL = None
    return config.SEMANTIC_MODEL


def _preprocess_df(df, is_training=True, field_types=None):
    """
    Preprocess DICOM header DataFrame with automatic field type detection.

    Args:
        df: DataFrame with DICOM header fields
        is_training: Whether this is training data (triggers type detection)
        field_types: Pre-computed field types (used during inference)

    Returns:
        If is_training: (preprocessed_df, categorical_cols, text_cols, field_types)
        If not is_training: preprocessed_df
    """
    field_names = [field[0] for field in DICOM_HEADER_FIELDS]

    # Select only the configured fields that exist in df
    available_fields = [f for f in field_names if f in df.columns]
    X = df[available_fields].copy()

    if is_training or field_types is None:
        logger.info("Analyzing DICOM header field types...")
        detected_field_types = analyze_dataframe_fields(X, available_fields)

        X_processed, categorical_cols, text_cols = preprocess_by_type(X, detected_field_types)

        if is_training:
            return X_processed, categorical_cols, text_cols, detected_field_types
        else:
            return X_processed
    else:
        X_processed, categorical_cols, text_cols = preprocess_by_type(X, field_types)
        return X_processed



def encode_text_features(feature_df, text_cols):
    embedded_features = []
    model = _get_semantic_model()
    if model is None:
        # If sentence-transformers isn't available, return zero-sized embedding frames to allow pipeline to continue.
        for col in text_cols:
            # Create empty DataFrame with zero columns matching expected usage
            emb_df = pd.DataFrame(index=feature_df.index)
            embedded_features.append(emb_df)
        return embedded_features

    for col in text_cols:
        if col not in feature_df.columns:
            emb_df = pd.DataFrame(index=feature_df.index)
            embedded_features.append(emb_df)
            continue

        text_data = feature_df[col].tolist()
        _t = time.perf_counter()   # [BENCH]
        embeddings = model.encode(
            text_data,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        logger.info(f"[BENCH]     text embed '{col}' ({len(text_data)} rows): "
                    f"{(time.perf_counter()-_t)*1000:.0f} ms")  # [BENCH]
        embedding_df = pd.DataFrame(
            embeddings,
            index=feature_df.index,
            columns=[f"{col}_{i}" for i in range(embeddings.shape[1])]
        )
        embedded_features.append(embedding_df)
    return embedded_features
