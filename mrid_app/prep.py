import re
import numpy as np
import pandas as pd
from .config import ARRAY_COLS_MAPPING, INITIAL_FEATURE_COLS, CATEGORICAL_COLS, TEXT_COLS, SEMANTIC_MODEL


def _get_semantic_model():
    global SEMANTIC_MODEL
    if SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            SEMANTIC_MODEL = None
    return SEMANTIC_MODEL


def _extract_and_clean_array_features(df):
    for col_name, size in ARRAY_COLS_MAPPING.items():
        if col_name in df.columns:
            def parse_array_string(val):
                if pd.isna(val) or val is None or val == 'None':
                    return [np.nan] * size
                try:
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
                    float_list = [float(n) for n in numbers[:size]]
                    float_list.extend([np.nan] * (size - len(float_list)))
                    return float_list[:size]
                except Exception:
                    return [np.nan] * size

            parsed_data = df[col_name].apply(parse_array_string).tolist()
            parsed_df = pd.DataFrame(parsed_data, index=df.index,
                                     columns=[f"{col_name}_{i}" for i in range(size)])

            df = df.drop(columns=[col_name])
            df = pd.concat([df, parsed_df], axis=1)

    if 'TI (Inversion Time)' in df.columns:
        df['TI (Inversion Time)'] = pd.to_numeric(df['TI (Inversion Time)'], errors='coerce')
    if 'SpacingBetweenSlices (mm)' in df.columns:
        df['SpacingBetweenSlices (mm)'] = pd.to_numeric(df['SpacingBetweenSlices (mm)'], errors='coerce')

    return df


def _preprocess_df(df, is_training=True):
    X = df[INITIAL_FEATURE_COLS].copy()
    X = _extract_and_clean_array_features(X)
    final_cols = X.columns.tolist()
    final_cat_cols = [c for c in final_cols if c in CATEGORICAL_COLS]
    final_text_cols = [c for c in final_cols if c in TEXT_COLS]
    final_num_cols = [c for c in final_cols if c not in final_cat_cols + final_text_cols]

    for col in final_num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1.0).astype(float)

    for col in final_cat_cols:
        X[col] = X[col].astype(str).fillna('MISSING').str.strip()

    for col in final_text_cols:
        X[col] = X[col].fillna('MISSING').astype(str).str.replace('_', ' ').str.strip()

    if is_training:
        return X, final_cat_cols, final_text_cols
    else:
        return X


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
