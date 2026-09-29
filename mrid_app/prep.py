import pandas as pd
from . import config
from .field_detection import analyze_dataframe_fields, preprocess_by_type
from .logging_config import get_logger

logger = get_logger()

#: The text embedding model of header models trained before MRID recorded it on
#: the model itself (the only one those versions used).
LEGACY_SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'

#: Attribute on a trained header model (RandomForestClassifier) naming the text
#: embedding model it was trained with; None if it was trained without one.
SEMANTIC_MODEL_ATTR = "mrid_semantic_model_name"

_loaded_models = {}       # model name -> loaded SentenceTransformer
_failed_models = {}       # model name -> error message (not retried every batch)
_warned = set()


class SemanticModelError(RuntimeError):
    """The configured sentence-transformers model could not be loaded."""


def configured_semantic_model_name():
    """SEMANTIC_MODEL_NAME from the active config (read at call time, so a
    project config's value applies)."""
    name = getattr(config, "SEMANTIC_MODEL_NAME", None)
    name = str(name).strip() if name is not None else ""
    return name or LEGACY_SEMANTIC_MODEL_NAME


def trained_semantic_model_name(rf_model):
    """The text embedding model *rf_model* was trained with (None: trained
    without text embeddings). Models trained before MRID recorded it used
    :data:`LEGACY_SEMANTIC_MODEL_NAME`."""
    return getattr(rf_model, SEMANTIC_MODEL_ATTR, LEGACY_SEMANTIC_MODEL_NAME)


def inference_semantic_model_name(rf_model):
    """The embedding model to classify with: always the one *rf_model* was
    trained with, since another model produces different features. Warns once
    if the config now names a different model."""
    trained = trained_semantic_model_name(rf_model)
    configured = configured_semantic_model_name()
    if trained is not None and trained != configured and (trained, configured) not in _warned:
        _warned.add((trained, configured))
        logger.warning(
            f"The header model was trained with the text embedding model '{trained}', so "
            f"it is used to classify. SEMANTIC_MODEL_NAME is '{configured}'; retrain the "
            f"header model to use it.")
    return trained


def get_semantic_model(name=None):
    """Load (once) and return the sentence-transformers model *name*, by default
    SEMANTIC_MODEL_NAME from the active config.

    *name* is anything ``SentenceTransformer()`` accepts: a model on the Hugging
    Face Hub (downloaded and cached on first use) or a local model folder.
    Returns None if sentence-transformers is not installed; header models then
    train without text embeddings. Raises :class:`SemanticModelError` if the
    model cannot be loaded, rather than silently training or classifying
    without the text features.
    """
    name = name or configured_semantic_model_name()
    if name in _loaded_models:
        return _loaded_models[name]
    if name in _failed_models:
        raise SemanticModelError(_failed_models[name])
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        if "import" not in _warned:
            _warned.add("import")
            logger.warning("sentence-transformers is not installed; the header model will "
                           "not use text embeddings of free-text fields (e.g. SeriesDescription).")
        return None

    import logging
    # Suppress noisy key-mismatch warnings from transformers during model load
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    prev_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    try:
        model = SentenceTransformer(name)
    except Exception as e:
        message = (
            f"Could not load the text embedding model '{name}' (SEMANTIC_MODEL_NAME): "
            f"{type(e).__name__}: {e}. Use the name of a sentence-transformers model on "
            f"the Hugging Face Hub (e.g. 'all-MiniLM-L6-v2' or 'all-mpnet-base-v2') or "
            f"the path to a local model folder. The first use downloads the model, so "
            f"the machine needs internet access unless the model is already cached.")
        _failed_models[name] = message
        logger.error(message)
        raise SemanticModelError(message) from e
    finally:
        transformers_logger.setLevel(prev_level)
    _loaded_models[name] = model
    logger.info(f"Loaded semantic model: {name}")
    return model


def _preprocess_df(df, is_training=True, field_types=None, field_names=None):
    """
    Preprocess DICOM header DataFrame with automatic field type detection.

    Args:
        df: DataFrame with DICOM header fields
        is_training: Whether this is training data (triggers type detection)
        field_types: Pre-computed field types (used during inference)
        field_names: Header field display names to use (default: those in
            DICOM_HEADER_FIELDS of the active config)

    Returns:
        If is_training: (preprocessed_df, categorical_cols, text_cols, field_types)
        If not is_training: preprocessed_df
    """
    if field_names is None:
        field_names = [field[0] for field in config.DICOM_HEADER_FIELDS]

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



def encode_text_features(feature_df, text_cols, model_name=None):
    """Embed each text column with the sentence-transformers model *model_name*
    (default: SEMANTIC_MODEL_NAME from the active config)."""
    embedded_features = []
    model = get_semantic_model(model_name) if text_cols else None
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
        embeddings = model.encode(
            text_data,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embedding_df = pd.DataFrame(
            embeddings,
            index=feature_df.index,
            columns=[f"{col}_{i}" for i in range(embeddings.shape[1])]
        )
        embedded_features.append(embedding_df)
    return embedded_features
