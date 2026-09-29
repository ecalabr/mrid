"""Settings that must follow the project's config.py, not the repository default.

- The headless CLI uses <output>/config.py when --config is not given.
- The scanner's ImageType filters and series checks come from the active config,
  including inside worker processes.
- SEMANTIC_MODEL_NAME chooses the header model's text embedding model, and a
  trained header model keeps classifying with the model it was trained with.
"""

import contextlib
import functools
import io
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import pandas as pd

from mrid_app import config
from mrid_app.config_io import get_default_config_path, replace_single_assignment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.contextmanager
def _preserve_config_module():
    saved = {n: getattr(config, n) for n in dir(config) if n.isupper()}
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(config, n, v)


@contextlib.contextmanager
def _tmpdir():
    path = tempfile.mkdtemp(prefix="mrid_ctl_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_dicom(path, image_type, study_uid, series_uid, desc="ax tee one"):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID = "P1"
    ds.AccessionNumber = "A1"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = desc
    ds.SeriesNumber = 1
    ds.Modality = "MR"
    ds.ImageType = image_type
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.Rows = ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.arange(64, dtype=np.uint16).reshape(8, 8).tobytes()
    ds.save_as(path)
    return path


def _image_type_files(folder):
    """One file per ImageType, each in its own series; returns {name: path}."""
    from pydicom.uid import generate_uid
    study = generate_uid()
    kinds = {
        "original": ["ORIGINAL", "PRIMARY", "M"],
        "derived": ["DERIVED", "SECONDARY", "M"],
        "localizer": ["ORIGINAL", "PRIMARY", "LOCALIZER"],
        "orig_secondary": ["ORIGINAL", "SECONDARY", "M"],
    }
    return {name: _write_dicom(os.path.join(folder, f"{name}.dcm"), it, study, generate_uid())
            for name, it in kinds.items()}


def _kept(files, results):
    by_path = {r["path"] for r in results if r}
    return sorted(name for name, p in files.items() if p in by_path)


# ── 1. headless uses the project's config ────────────────────────────────────

def _headless():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import headless_autoclassify
    return headless_autoclassify


def test_resolve_config_prefers_explicit_then_output_then_default():
    h = _headless()
    with _tmpdir() as out:
        assert h.resolve_config_path(None, out) is None
        with open(os.path.join(out, "config.py"), "w") as fh:
            fh.write("# project\n")
        assert h.resolve_config_path(None, out) == os.path.join(out, "config.py")
        assert h.resolve_config_path("/x/other.py", out) == "/x/other.py"
        os.remove(os.path.join(out, "config.py"))
        os.makedirs(os.path.join(out, "config.py"))  # a folder is not a config
        assert h.resolve_config_path(None, out) is None


def _run(inp, out, *extra):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "headless_autoclassify.py"), "-i", inp, "-o", out, *extra],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=REPO, timeout=300)


def _write_project_config(out, **settings):
    with open(get_default_config_path(), encoding="utf-8") as fh:
        source = fh.read()
    for name, value in settings.items():
        source = replace_single_assignment(source, name, repr(value))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "config.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    return path


def test_headless_uses_the_output_config_and_its_image_type_filter():
    """End to end: the project config in -o is loaded without --config, and its
    INCLUDE_TYPES reaches the scan workers."""
    with _tmpdir() as tmp:
        inp = os.path.join(tmp, "in")
        os.makedirs(inp)
        _image_type_files(inp)

        default_out = os.path.join(tmp, "default_out")
        os.makedirs(default_out)
        # The runs end at "RF model not found" (no trained model in these
        # projects); what matters here is everything before that.
        proc = _run(inp, default_out)
        log = proc.stdout + proc.stderr
        assert "RF model not found" in log, log[-2000:]
        assert "No config.py found in the output directory" in log
        assert "Config:     repository default" in log
        assert "Identified 1 valid DICOM files" in log  # only ORIGINAL\PRIMARY, no LOCALIZER

        project_out = os.path.join(tmp, "project_out")
        cfg = _write_project_config(project_out, INCLUDE_TYPES=["ORIGINAL"], EXCLUDE_TYPES=[])
        proc = _run(inp, project_out)
        log = proc.stdout + proc.stderr
        assert "RF model not found" in log, log[-2000:]
        assert f"Config:     {cfg}" in log
        assert "No config.py found" not in log
        assert "Identified 3 valid DICOM files" in log  # every ORIGINAL file


def test_headless_output_config_gets_the_missing_settings_check():
    with _tmpdir() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp)
        cfg = _write_project_config(out)
        with open(cfg, encoding="utf-8") as fh:
            source = fh.read()
        with open(cfg, "w", encoding="utf-8") as fh:
            fh.write(source.replace("FUSION_METHOD = 'reliability'\n", ""))
        proc = _run(inp, out)
        assert proc.returncode == 1, proc.stderr[-2000:]
        assert f"copy these lines into {cfg}" in proc.stderr
        assert "FUSION_METHOD = 'reliability'" in proc.stderr
        proc = _run(inp, out, "--accept-config-defaults")
        assert proc.returncode == 0, proc.stderr[-2000:]
        with open(cfg, encoding="utf-8") as fh:
            assert "FUSION_METHOD = 'reliability'" in fh.read()


# ── 2. scanner settings come from the active config ──────────────────────────

def test_image_type_filters_follow_the_config():
    from mrid_app.scanner import check_dicom_batch
    with _tmpdir() as tmp, _preserve_config_module():
        files = _image_type_files(tmp)
        paths = list(files.values())
        assert _kept(files, check_dicom_batch(paths)) == ["original"]

        config.INCLUDE_TYPES = ["ORIGINAL"]
        config.EXCLUDE_TYPES = ["LOCALIZER"]
        assert _kept(files, check_dicom_batch(paths)) == ["orig_secondary", "original"]

        config.INCLUDE_TYPES = []  # no requirement (used to fall back to the defaults)
        config.EXCLUDE_TYPES = []
        assert _kept(files, check_dicom_batch(paths)) == sorted(files)

        config.INCLUDE_TYPES = ["original", "primary"]  # case does not matter
        config.EXCLUDE_TYPES = ["localizer"]
        assert _kept(files, check_dicom_batch(paths)) == ["original"]


def test_explicit_options_override_the_config():
    from mrid_app.scanner import check_dicom_batch
    with _tmpdir() as tmp:
        files = _image_type_files(tmp)
        kept = check_dicom_batch(list(files.values()), include_types=["DERIVED"], exclude_types=[])
        assert _kept(files, kept) == ["derived"]


def test_scan_options_reach_spawned_worker_processes():
    """A spawned worker re-imports the repository config; the project's settings
    only arrive if they are passed in explicitly, as the callers now do."""
    from mrid_app.scanner import check_dicom_batch, scan_options
    with _tmpdir() as tmp, _preserve_config_module():
        files = _image_type_files(tmp)
        paths = list(files.values())
        config.INCLUDE_TYPES = ["DERIVED"]
        config.EXCLUDE_TYPES = []
        ctx = mp.get_context("spawn")
        with ctx.Pool(1) as pool:
            unconfigured = pool.map(check_dicom_batch, [paths])[0]
            configured = pool.map(functools.partial(check_dicom_batch, **scan_options()), [paths])[0]
        assert _kept(files, unconfigured) == ["original"], "premise: the worker sees defaults"
        assert _kept(files, configured) == ["derived"]


def test_series_checks_follow_the_config():
    from mrid_app.scanner import build_hierarchy

    def records():
        base = {"patient": "P", "study_uid": "S", "accession_number": "A",
                "series_uid": "SER", "series_desc": "mixed", "series_number": 1}
        return [dict(base, path=f"/fake/{i}.dcm", image_position=[0, 0, float(i)],
                     image_orientation=o)
                for i, o in enumerate([[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, -1]])]

    with _tmpdir() as tmp, _preserve_config_module():
        config.IGNORE_MULTIPLE_ORIENTATIONS = True
        assert build_hierarchy(records(), output_dir=tmp) == {}
        config.IGNORE_MULTIPLE_ORIENTATIONS = False
        assert "P" in build_hierarchy(records(), output_dir=tmp)


# ── 4. the text embedding model is configurable ──────────────────────────────

class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer: deterministic
    embeddings whose width depends on the model, so using the wrong model at
    inference shows up as a feature-count mismatch."""
    loads = []
    dims = {"fake-small": 4, "fake-large": 6, "all-MiniLM-L6-v2": 5}

    def __init__(self, name):
        _FakeSentenceTransformer.loads.append(name)
        if name not in self.dims:
            raise OSError(f"{name} is not a valid model identifier")
        self.dim = self.dims[name]

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        return np.array([[(hash(t) % 97 + k) / 97.0 for k in range(self.dim)] for t in texts])


@contextlib.contextmanager
def _fake_sentence_transformers(installed=True):
    from mrid_app import prep
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeSentenceTransformer
    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake if installed else None
    _FakeSentenceTransformer.loads = []
    caches = (prep._loaded_models, prep._failed_models, prep._warned)
    for c in caches:
        c.clear()
    try:
        yield
    finally:
        for c in caches:
            c.clear()
        if saved is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = saved


@contextlib.contextmanager
def _captured_warnings():
    from mrid_app.logging_config import get_logger
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    logger = get_logger()
    logger.addHandler(handler)
    try:
        yield buf
    finally:
        logger.removeHandler(handler)


class _StubGui:
    def __init__(self):
        self.statuses = []

    def update_status(self, text):
        self.statuses.append(text)

    def is_cancelled(self):
        return False

    def close(self):
        pass

    def __getattr__(self, name):  # any other progress hook is a no-op
        return lambda *a, **k: None


def _training_frame():
    rng = np.random.RandomState(0)
    words = {"T1": "axial tee one", "T2": "axial tee two", "FLAIR": "fluid attenuated axial"}
    rows = []
    for label, desc in words.items():
        for i in range(12):
            rows.append({"Classification": label, "Path": f"/x/{label}{i}.dcm",
                         "SeriesDescription": f"{desc} {['pre', 'post', 'fs'][i % 3]}",
                         "TR (Repetition Time)": float(rng.randint(400, 9000))})
    return pd.DataFrame(rows)


def _train_header(out):
    from mrid_app import header
    saved_time = header.time
    header.time = types.SimpleNamespace(sleep=lambda *_: None, time=saved_time.time)
    gui = _StubGui()
    try:
        header.TrainHeaderClassifierLogic(out, gui, preloaded_df=_training_frame())
    finally:
        header.time = saved_time
    path = os.path.join(out, "model_files", "rf_classifier_model.joblib")
    return (path if os.path.exists(path) else None), gui


_EXTRACTED_FIELDS = []


def _classify(out, frame):
    """Classify *frame*'s rows with the header model saved in *out*, reading
    "DICOM headers" from the frame (any field it lacks reads as missing)."""
    import joblib
    from mrid_app import header
    model = joblib.load(os.path.join(out, "model_files", "rf_classifier_model.joblib"))
    encoder = joblib.load(os.path.join(out, "model_files", "feature_encoder.joblib"))
    field_types = joblib.load(os.path.join(out, "model_files", "field_types.joblib"))
    features = {row["Path"]: row.drop(["Classification", "Path"]).to_dict()
                for _, row in frame.iterrows()}

    def fake_features(path, fields=None):
        _EXTRACTED_FIELDS.append(list(fields))
        return {name: features[path].get(name, "None") for name, _ in fields}

    saved = header.get_series_features
    header.get_series_features = fake_features
    _EXTRACTED_FIELDS.clear()
    try:
        return model, header.get_series_probabilities_batch(list(features), model, encoder, field_types)
    finally:
        header.get_series_features = saved


def test_configured_model_name_is_read_at_call_time():
    from mrid_app.prep import configured_semantic_model_name
    with _preserve_config_module():
        config.SEMANTIC_MODEL_NAME = "fake-large"
        assert configured_semantic_model_name() == "fake-large"
        config.SEMANTIC_MODEL_NAME = "  "
        assert configured_semantic_model_name() == "all-MiniLM-L6-v2"


def test_header_model_trains_and_classifies_with_the_configured_model():
    with _fake_sentence_transformers(), _preserve_config_module(), _tmpdir() as out:
        config.SEMANTIC_MODEL_NAME = "fake-large"
        path, gui = _train_header(out)
        assert path, gui.statuses
        model, probas = _classify(out, _training_frame())
        assert model.mrid_semantic_model_name == "fake-large"
        assert model.n_features_in_ == 1 + 6, "TR plus the 6-wide fake-large embedding"
        assert all(p is not None for p in probas)
        assert _FakeSentenceTransformer.loads == ["fake-large"], "loaded once and reused"


def test_changing_the_model_keeps_classifying_with_the_trained_one():
    with _fake_sentence_transformers(), _preserve_config_module(), _tmpdir() as out:
        config.SEMANTIC_MODEL_NAME = "fake-large"
        config.DICOM_HEADER_FIELDS = [("SeriesDescription", "SeriesDescription"),
                                      ("TR (Repetition Time)", "RepetitionTime")]
        assert _train_header(out)[0]
        config.SEMANTIC_MODEL_NAME = "fake-small"  # changed after training
        with _captured_warnings() as warnings:
            _, probas = _classify(out, _training_frame())
            _classify(out, _training_frame())
        assert all(p is not None for p in probas), "the trained model's features must be used"
        assert "fake-small" not in _FakeSentenceTransformer.loads
        assert warnings.getvalue().count("retrain the header model") == 1, warnings.getvalue()
        assert "SEMANTIC_MODEL_NAME is 'fake-small'" in warnings.getvalue()


def test_legacy_header_models_use_the_historical_model():
    from mrid_app.prep import inference_semantic_model_name, trained_semantic_model_name
    legacy = types.SimpleNamespace()
    assert trained_semantic_model_name(legacy) == "all-MiniLM-L6-v2"
    without = types.SimpleNamespace(mrid_semantic_model_name=None)
    with _fake_sentence_transformers(), _preserve_config_module():
        config.SEMANTIC_MODEL_NAME = "fake-small"
        assert inference_semantic_model_name(without) is None


def test_an_unloadable_model_stops_training_with_guidance():
    from mrid_app.prep import SemanticModelError, get_semantic_model
    with _fake_sentence_transformers(), _preserve_config_module(), _tmpdir() as out:
        config.SEMANTIC_MODEL_NAME = "no-such-model"
        path, gui = _train_header(out)
        assert path is None, "no header model may be saved without its text features"
        final = gui.statuses[-1]
        assert "Could not load the text embedding model 'no-such-model'" in final
        assert "Hugging Face Hub" in final
        try:
            get_semantic_model("no-such-model")
        except SemanticModelError:
            pass
        assert _FakeSentenceTransformer.loads == ["no-such-model"], "failures are not retried"


def test_without_sentence_transformers_training_and_classification_still_work():
    with _fake_sentence_transformers(installed=False), _preserve_config_module(), _tmpdir() as out:
        path, gui = _train_header(out)
        assert path, gui.statuses
        model, probas = _classify(out, _training_frame())
        assert model.mrid_semantic_model_name is None
        assert all(p is not None for p in probas)


def test_the_real_default_model_loads_if_cached():
    """The shipped default must load through the new path (skipped offline)."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return
    from mrid_app import prep
    if not any("MiniLM-L6-v2" in d for root in (os.path.expanduser("~/.cache/huggingface"),)
               for d, _, _ in os.walk(root)):
        return
    model = prep.get_semantic_model("all-MiniLM-L6-v2")
    assert model is not None
    assert model.encode(["axial t1 post"], convert_to_numpy=True).shape == (1, 384)


# ── header fields and tree count come from the active config ────────────────

_TRAINED_FIELDS = [("SeriesDescription", "SeriesDescription"),
                   ("TR (Repetition Time)", "RepetitionTime")]


def test_tree_count_follows_the_config():
    with _fake_sentence_transformers(), _preserve_config_module(), _tmpdir() as out:
        config.SEMANTIC_MODEL_NAME = "fake-small"
        config.N_ESTIMATORS = 7
        path, gui = _train_header(out)
        import joblib
        assert joblib.load(path).n_estimators == 7
        assert any("(7 trees)" in status for status in gui.statuses)


def test_header_fields_follow_the_project_config():
    with _fake_sentence_transformers(), _preserve_config_module(), _tmpdir() as out:
        config.SEMANTIC_MODEL_NAME = "fake-large"
        config.DICOM_HEADER_FIELDS = [("SeriesDescription", "SeriesDescription")]
        assert _train_header(out)[0]
        model, probas = _classify(out, _training_frame())
        assert model.mrid_header_fields == [("SeriesDescription", "SeriesDescription")]
        assert model.n_features_in_ == 6, "only the embedding; TR is not a configured field"
        assert all(p is not None for p in probas)


def test_trained_fields_are_recorded_and_used_after_the_config_changes():
    with _fake_sentence_transformers(), _preserve_config_module(), _tmpdir() as out:
        config.SEMANTIC_MODEL_NAME = "fake-small"
        assert _train_header(out)[0]  # default list: the frame has SeriesDescription and TR
        config.DICOM_HEADER_FIELDS = [("SeriesDescription", "SeriesDescription"),
                                      ("TE (Echo Time)", "EchoTime")]
        with _captured_warnings() as warnings:
            model, probas = _classify(out, _training_frame())
            _classify(out, _training_frame())
        assert model.mrid_header_fields == _TRAINED_FIELDS
        assert _EXTRACTED_FIELDS and all(f == _TRAINED_FIELDS for f in _EXTRACTED_FIELDS), \
            "classification must read the fields the model was trained on"
        assert all(p is not None for p in probas)
        text = warnings.getvalue()
        assert text.count("retrain the header model to use the configured list") == 1, text
        assert "not in the model: TE (Echo Time)" in text
        assert "no longer configured: TR (Repetition Time)" in text


def test_older_models_take_their_fields_from_the_saved_field_types():
    from mrid_app.header import trained_header_fields
    legacy = types.SimpleNamespace()
    field_types = {"SeriesDescription": {"type": "text"}, "TR (Repetition Time)": {"type": "numerical"}}
    with _preserve_config_module():
        # A project list without TR: its DICOM attribute comes from the repository list.
        config.DICOM_HEADER_FIELDS = [("SeriesDescription", "SeriesDescription")]
        assert trained_header_fields(legacy, field_types) == _TRAINED_FIELDS
        assert trained_header_fields(None, field_types) == config.DICOM_HEADER_FIELDS
        assert trained_header_fields(legacy, None) == config.DICOM_HEADER_FIELDS


def test_classification_csvs_and_backfill_use_the_configured_fields():
    from mrid_app import csv_field_updater, processing
    import csv as csv_module
    from pydicom.uid import generate_uid
    fields = [("SeriesDescription", "SeriesDescription"), ("Modality", "Modality")]
    with _preserve_config_module(), _tmpdir() as tmp:
        config.DICOM_HEADER_FIELDS = fields
        assert csv_field_updater.get_expected_field_names() == ["SeriesDescription", "Modality"]
        dcm = _write_dicom(os.path.join(tmp, "a.dcm"), ["ORIGINAL", "PRIMARY"],
                           generate_uid(), generate_uid(), desc="sag tee two")
        assert csv_field_updater.extract_dicom_field_value(dcm, "Modality") == "MR"
        processing.background_conversion_worker(
            "MRN", "UID", tmp, {dcm: "T2"}, {"sag tee two": [{"dicom_path": dcm}]}, "ACC",
            convert=False)
        written = [f for root, _, files in os.walk(tmp) for f in files if f.endswith(".csv")]
        assert written, "no classification CSV was written"
        with open(os.path.join([r for r, _, fs in os.walk(tmp) if written[0] in fs][0], written[0])) as fh:
            rows = list(csv_module.reader(fh))
        assert rows[0] == ["Classification", "Path", "SeriesDescription", "Modality"]
        assert rows[1][0] == "T2" and rows[1][2:] == ["sag tee two", "MR"]
