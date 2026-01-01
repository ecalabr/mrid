import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pydicom
from .prep import _preprocess_df, encode_text_features
from .config import N_ESTIMATORS, CATEGORICAL_COLS, TEXT_COLS


def TrainHeaderClassifierLogic(output_dir, gui):
    if not all([pd, RandomForestClassifier, joblib, LabelEncoder, OrdinalEncoder]):
        gui.update_status("❌ Error: Missing required dependencies for Header Model.")
        time.sleep(2)
        gui.close()
        return

    try:
        gui.update_status("1/6: Aggregating classification CSVs for Header Model...")
        all_data = []
        for study_folder in os.listdir(output_dir):
            study_path = os.path.join(output_dir, study_folder)
            if os.path.isdir(study_path):
                study_id = os.path.basename(study_path.rstrip(os.sep))
                csv_path = os.path.join(study_path, f"{study_id}_classification-results.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

        if not all_data:
            gui.update_status("❌ Error: No classification CSV files found to train on.")
            time.sleep(2)
            gui.close()
            return

        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv(os.path.join(output_dir, "classification_data.csv"), index=False)

        gui.update_status("2/6: Preprocessing and expanding features...")
        master_df = master_df[master_df['Classification'].str.lower() != 'none'].copy()
        master_df = master_df[master_df['Classification'].str.lower() != 'other'].copy()
        if master_df.empty:
            gui.update_status("❌ Error: Training data is empty after removing 'None'/'other' labels.")
            time.sleep(2)
            gui.close()
            return

        X_clean, categorical_cols, text_cols = _preprocess_df(master_df.drop(columns=['Classification', 'Path']), is_training=True)
        y = master_df['Classification']

        gui.update_status("3/6: Encoding categorical features and target...")

        target_encoder_filename = os.path.join(output_dir, "label_encoder.joblib")
        if os.path.exists(target_encoder_filename):
            target_encoder = joblib.load(target_encoder_filename)
            y_encoded = target_encoder.transform(y)
        else:
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)

        feature_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded_cat = feature_encoder.fit_transform(X_clean[categorical_cols])
        embedded_features_dfs = encode_text_features(X_clean, text_cols)

        X_final_num = X_clean.drop(columns=categorical_cols + text_cols).copy()
        X_encoded_cat_df = pd.DataFrame(X_encoded_cat, index=X_final_num.index, columns=categorical_cols)
        X_final = pd.concat([X_final_num, X_encoded_cat_df] + embedded_features_dfs, axis=1)

        gui.update_status("4/6: Training RandomForestClassifier (100 trees)...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)

        gui.update_status("5/6: Saving model, target encoder, and feature encoder...")
        model_filename = os.path.join(output_dir, "rf_classifier_model.joblib")
        feature_encoder_filename = os.path.join(output_dir, "feature_encoder.joblib")

        joblib.dump(model, model_filename)
        joblib.dump(target_encoder, target_encoder_filename)
        joblib.dump(feature_encoder, feature_encoder_filename)

        gui.update_status("6/6: Evaluating model and saving results...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        final_message = f"✅ Header Model Training Complete!\nAccuracy on Test Set: {accuracy:.2f}\nModel and Encoders Saved."
        summary_path = os.path.join(output_dir, "rf_model_training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(final_message)

        gui.update_status(final_message)
        time.sleep(5)

    except Exception as e:
        error_msg = f"❌ An unexpected error occurred during Header Model training: {type(e).__name__}: {str(e)}"
        print(error_msg)
        gui.update_status(error_msg)
        time.sleep(3)

    finally:
        gui.close()


# Helper: safely extract attribute values
def safe_get_dicom_attribute(ds, tag_name):
    value = getattr(ds, tag_name, None)
    if isinstance(value, pydicom.multival.MultiValue):
        value = str(tuple(value))
    return value if value is not None else "None"


def get_series_features(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
    except Exception:
        return None

    attr_map = {
        "SeriesDescription": "SeriesDescription",
        "TR (Repetition Time)": "RepetitionTime",
        "TE (Echo Time)": "EchoTime",
        "TI (Inversion Time)": "InversionTime",
        "ScanningSequence": "ScanningSequence",
        "SequenceVariant": "SequenceVariant",
        "ScanOptions": "ScanOptions",
        "SequenceName": "SequenceName",
        "AcquisitionType": "AcquisitionType",
        "ImageType": "ImageType",
        "Rows": "Rows",
        "Columns": "Columns",
        "PixelSpacing (mm)": "PixelSpacing",
        "SliceThickness (mm)": "SliceThickness",
        "ImagePositionPatient (mm)": "ImagePositionPatient",
        "ImageOrientationPatient": "ImageOrientationPatient",
        "SliceLocation (mm)": "SliceLocation",
        "SpacingBetweenSlices (mm)": "SpacingBetweenSlices",
        "ContrastBolusAgent": "ContrastBolusAgent"
    }

    features = {}
    for col in [
        "SeriesDescription", "TR (Repetition Time)", "TE (Echo Time)", "TI (Inversion Time)",
        "ScanningSequence", "SequenceVariant", "ScanOptions", "SequenceName",
        "AcquisitionType", "ImageType", "Rows", "Columns", "PixelSpacing (mm)",
        "SliceThickness (mm)", "ImagePositionPatient (mm)", "ImageOrientationPatient",
        "SliceLocation (mm)", "SpacingBetweenSlices (mm)",
        "ContrastBolusAgent"]:
        dicom_attr = attr_map.get(col, col)
        features[col] = safe_get_dicom_attribute(ds, dicom_attr)

    return features


def get_series_probabilities(dicom_path, model, feature_encoder, *args, **kwargs):
    """Predict probability scores for a DICOM series using the trained RF header model.

    Accepts optional extra arguments for backward compatibility (they are ignored).
    """
    if not all([model, feature_encoder]):
        return None
    features = get_series_features(dicom_path)
    if not features:
        return None

    X_pred_raw = pd.DataFrame([features])
    X_clean = _preprocess_df(X_pred_raw, is_training=False)

    categorical_cols = [c for c in X_clean.columns if c in CATEGORICAL_COLS]
    text_cols = [c for c in X_clean.columns if c in TEXT_COLS]

    try:
        X_encoded_cat = feature_encoder.transform(X_clean[categorical_cols]) if categorical_cols else []
        embedded_features_dfs = encode_text_features(X_clean, text_cols) if text_cols else []
        X_final_num = X_clean.drop(columns=categorical_cols + text_cols) if categorical_cols or text_cols else X_clean
        if len(X_encoded_cat) > 0:
            X_encoded_cat_df = pd.DataFrame(X_encoded_cat, index=X_final_num.index, columns=categorical_cols)
            X_final = pd.concat([X_final_num, X_encoded_cat_df] + embedded_features_dfs, axis=1)
        else:
            X_final = pd.concat([X_final_num] + embedded_features_dfs, axis=1)

        y_pred_proba = model.predict_proba(X_final)[0]
        return y_pred_proba
    except Exception as e:
        print(f"RF Prediction probability failed: {e}")
        return None
