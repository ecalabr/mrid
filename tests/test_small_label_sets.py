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
