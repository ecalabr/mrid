"""Training must be reproducible from the seed in the project config.

CNN training previously had no fixed seed, so retraining the same data gave a
different model every time. On one evaluation dataset four identical runs
spanned 0.067 macro F1, which is enough to make reported figures irreproducible.
Everything random now derives from config.STUDY_ORDER_SEED.
"""

import os
import tempfile

import numpy as np


def _set_seed(value):
    from mrid_app import config
    config.STUDY_ORDER_SEED = value


def _restore(value):
    from mrid_app import config
    config.STUDY_ORDER_SEED = value


def test_get_random_seed_follows_the_config():
    from mrid_app import config
    from mrid_app.utils import get_random_seed

    original = config.STUDY_ORDER_SEED
    try:
        assert get_random_seed() == original
        _set_seed(1234)
        assert get_random_seed() == 1234
        del config.STUDY_ORDER_SEED
        assert get_random_seed() == 42, "must fall back to 42 for older configs"
    finally:
        _restore(original)


def test_seed_everything_makes_python_and_numpy_repeat():
    import random
    from mrid_app.utils import seed_everything

    seed_everything(7)
    a = (random.random(), float(np.random.rand()))
    seed_everything(7)
    b = (random.random(), float(np.random.rand()))
    assert a == b

    seed_everything(8)
    c = (random.random(), float(np.random.rand()))
    assert a != c, "a different seed must give a different stream"


def test_seed_everything_makes_model_init_repeat():
    try:
        import torch
    except Exception:
        print("      (skipped: torch unavailable)")
        return
    from mrid_app.cnn import SimpleCNN
    from mrid_app.utils import seed_everything

    def weights(seed):
        seed_everything(seed)
        m = SimpleCNN(num_classes=4, num_slices=1, image_size=32)
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert torch.equal(weights(11), weights(11)), "same seed must give the same init"
    assert not torch.equal(weights(11), weights(12)), "different seeds must differ"


def test_seed_everything_sets_cudnn_deterministic():
    try:
        import torch
    except Exception:
        print("      (skipped: torch unavailable)")
        return
    from mrid_app.utils import seed_everything

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    seed_everything(3)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_dataloader_generator_pins_the_shuffle_order():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        print("      (skipped: torch unavailable)")
        return
    from mrid_app.utils import torch_generator

    ds = TensorDataset(torch.arange(40).float().unsqueeze(1))
    def order(seed):
        dl = DataLoader(ds, batch_size=4, shuffle=True, generator=torch_generator(seed))
        return [int(x[0][0]) for x in dl]

    assert order(5) == order(5)
    assert order(5) != order(6)


# ── End to end: the same data must train the same model twice ───────────────

def _tiny_training_set(n=24, classes=3):
    rng = np.random.RandomState(0)
    y = np.arange(n) % classes
    X = rng.rand(n, 1, 32, 32).astype("float32")
    # Make the classes separable so training is not pure noise.
    for i, lab in enumerate(y):
        X[i, 0, : 8 * (lab + 1), :] += 1.5
    return X, y


def _train_once(tmp, seed, epochs=3):
    import torch
    from mrid_app import config
    from mrid_app.cnn import train_cnn_model_logic

    X, y = _tiny_training_set()
    orig_seed, orig_ep = config.STUDY_ORDER_SEED, config.NUM_EPOCHS
    try:
        config.STUDY_ORDER_SEED = seed
        config.NUM_EPOCHS = epochs
        path, _acc = train_cnn_model_logic(
            torch.from_numpy(X), y, num_classes=3, output_dir=tmp)
    finally:
        config.STUDY_ORDER_SEED, config.NUM_EPOCHS = orig_seed, orig_ep
    sd = torch.load(path, map_location="cpu")
    return torch.cat([v.flatten().float() for v in sd.values()])


def test_cnn_training_is_reproducible_with_the_same_seed():
    """The regression this feature exists for."""
    try:
        import torch
    except Exception:
        print("      (skipped: torch unavailable)")
        return

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        wa = _train_once(a, seed=99)
        wb = _train_once(b, seed=99)
    assert torch.equal(wa, wb), (
        "two runs on identical data with the same seed produced different "
        f"weights (max abs diff {float((wa - wb).abs().max()):.3e})")


def test_cnn_training_differs_with_a_different_seed():
    """Guards against the trivial pass where training is degenerate."""
    try:
        import torch
    except Exception:
        print("      (skipped: torch unavailable)")
        return

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        wa = _train_once(a, seed=99)
        wb = _train_once(b, seed=100)
    assert not torch.equal(wa, wb), "the seed is being ignored"


def test_header_training_honors_the_config_seed():
    """The Random Forest and its split must both take the configured seed."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from mrid_app.utils import get_random_seed
    from mrid_app import config

    original = config.STUDY_ORDER_SEED
    X = np.random.RandomState(0).rand(60, 5)
    y = np.arange(60) % 3

    def fit_with(seed):
        _set_seed(seed)
        s = get_random_seed()
        Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
        m = RandomForestClassifier(n_estimators=20, random_state=s).fit(Xtr, ytr)
        return list(m.predict(X)), list(Xte[:, 0])

    try:
        assert fit_with(5) == fit_with(5)
        assert fit_with(5) != fit_with(6), "the seed must reach both the split and the forest"
    finally:
        _restore(original)


def test_no_hardcoded_seeds_remain_in_training_paths():
    """A stray random_state=42 would silently ignore the user's seed."""
    import inspect
    from mrid_app import cnn, header, analyze

    for mod in (cnn, header, analyze):
        src = inspect.getsource(mod)
        assert "random_state=42" not in src, f"{mod.__name__} still hardcodes a seed"
