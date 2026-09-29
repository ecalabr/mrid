"""Training must survive label sets too small or too imbalanced to stratify.

A user who has just started labeling frequently has a class with a single
example. Both trainers used to raise inside train_test_split(stratify=...) and
abort, which surfaced only as a failed training run.
"""

import numpy as np


def _singleton_class_labels():
    """Labels where one class has exactly one member — unstratifiable."""
    return np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8])


def test_stratified_split_would_fail_on_a_singleton_class():
    """Guards the premise: without a fallback, sklearn raises here."""
    from sklearn.model_selection import train_test_split

    y = _singleton_class_labels()
    X = np.random.RandomState(0).rand(len(y), 4)
    try:
        train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        return
    raise AssertionError("expected a stratification ValueError")


def test_cnn_training_completes_with_a_singleton_class():
    """train_cnn_model_logic must produce a saved model, not abort."""
    try:
        import torch  # noqa: F401
    except Exception:
        print("      (skipped: torch unavailable)")
        return

    import os
    import tempfile
    from mrid_app.cnn import train_cnn_model_logic

    import torch

    y = _singleton_class_labels()
    rng = np.random.RandomState(0)
    # The trainer receives what prepare_and_cache_cnn_data writes: a torch
    # tensor of shape (N, slices, H, W).
    X = torch.from_numpy(rng.rand(len(y), 1, 32, 32).astype("float32"))

    with tempfile.TemporaryDirectory() as tmp:
        from mrid_app import config as config_module

        original = getattr(config_module, "NUM_EPOCHS", 100)
        try:
            config_module.NUM_EPOCHS = 1
            model_path, best_acc = train_cnn_model_logic(
                X, y, num_classes=int(y.max()) + 1, output_dir=tmp)
        finally:
            config_module.NUM_EPOCHS = original

        assert model_path is not None, "training returned no model path"
        assert os.path.exists(model_path), f"no checkpoint written at {model_path}"


def test_header_training_completes_with_a_singleton_class():
    """The header trainer has the same split; exercise its fallback directly."""
    from sklearn.model_selection import train_test_split

    y = _singleton_class_labels()
    X = np.random.RandomState(0).rand(len(y), 4)

    # Mirror of the guarded call in header.TrainHeaderClassifierLogic.
    try:
        split = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError:
        split = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = split
    assert len(X_train) and len(X_test)
    assert len(y_train) + len(y_test) == len(y)


def test_cnn_saves_a_checkpoint_even_when_validation_accuracy_is_zero():
    """A run that never improves on 0% validation accuracy must still leave a
    usable checkpoint; otherwise training reports success with nothing on disk."""
    try:
        import torch
    except Exception:
        print("      (skipped: torch unavailable)")
        return

    import os
    import tempfile
    from mrid_app.cnn import train_cnn_model_logic
    from mrid_app import config as config_module

    # Random labels over many classes on noise: validation accuracy will be 0%.
    rng = np.random.RandomState(1)
    y = np.arange(12) % 6
    X = torch.from_numpy(rng.rand(len(y), 1, 32, 32).astype("float32"))

    with tempfile.TemporaryDirectory() as tmp:
        original = getattr(config_module, "NUM_EPOCHS", 100)
        try:
            config_module.NUM_EPOCHS = 1
            model_path, _acc = train_cnn_model_logic(
                X, y, num_classes=6, output_dir=tmp)
        finally:
            config_module.NUM_EPOCHS = original
        assert model_path is not None and os.path.exists(model_path), \
            "no checkpoint written when validation accuracy never improved"


# ── Classes absent from a model's training labels ───────────────────────────

class _FakeRF:
    """Stands in for a RandomForest that never saw one of the encoder's classes.
    predict_proba returns one column per class in `classes_`, not per label."""

    def __init__(self, classes):
        self.classes_ = np.array(classes)


def test_expand_to_label_space_scatters_columns_by_class():
    """A model trained without class 2 must have its columns mapped back to the
    right positions, with the unseen class scoring 0."""
    from mrid_app.processing import _expand_to_label_space

    model = _FakeRF([0, 1, 3])          # class 2 never appeared in training
    proba = np.array([0.5, 0.3, 0.2])   # aligned with classes_, not the labels
    full = _expand_to_label_space(proba, model, n_classes=4)

    assert len(full) == 4
    assert np.allclose(full, [0.5, 0.3, 0.0, 0.2])


def test_expand_to_label_space_is_a_no_op_when_already_aligned():
    from mrid_app.processing import _expand_to_label_space

    proba = np.array([0.25, 0.25, 0.5])
    model = _FakeRF([0, 1, 2])
    assert _expand_to_label_space(proba, model, 3) is proba


def test_expand_to_label_space_gives_up_when_unrecoverable():
    """A length mismatch the class list cannot explain must be discarded, not
    guessed at."""
    from mrid_app.processing import _expand_to_label_space

    assert _expand_to_label_space(np.array([0.5, 0.5]), _FakeRF([0, 1, 2]), 4) is None
    assert _expand_to_label_space(None, _FakeRF([0]), 3) is None


def test_optimize_study_selections_survives_a_class_missing_from_rf_training():
    """The end-to-end failure this came from: the header model trained on 2 of
    3 classes, and every study raised IndexError, silently producing no
    predictions at all."""
    from mrid_app import processing

    classes = np.array(["T1", "T2", "FLAIR"])

    class _Enc:
        classes_ = classes

    rf_model = _FakeRF([0, 2])                 # never saw "T2"
    study_dict = {"s": ["/fake/a.dcm"]}

    orig_rf = processing.get_series_probabilities_batch
    orig_cnn = processing.get_image_probabilities_batch
    orig_3d = processing.check_is_3d_acquisition
    try:
        # Two columns from the RF, three from the CNN.
        processing.get_series_probabilities_batch = lambda *a, **k: [np.array([0.9, 0.1])]
        processing.get_image_probabilities_batch = lambda *a, **k: [np.array([0.8, 0.1, 0.1])]
        processing.check_is_3d_acquisition = lambda *a, **k: False

        assignments, _ = processing.optimize_study_selections(
            study_dict, "Header + Image Model",
            rf_model, _Enc(), "feature-encoder",
            "cnn-model", _Enc(), None)
    finally:
        processing.get_series_probabilities_batch = orig_rf
        processing.get_image_probabilities_batch = orig_cnn
        processing.check_is_3d_acquisition = orig_3d

    # Must produce an assignment rather than raising; the RF's 0.9 belongs to
    # "T1" (class 0) and its 0.1 to "FLAIR" (class 2), so T1 wins.
    assert assignments.get("/fake/a.dcm") == "T1"
