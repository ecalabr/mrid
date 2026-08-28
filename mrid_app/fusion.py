"""Fusion rules for combining the DICOM-header (Random Forest) and image (CNN)
classifiers into a single per-series probability vector.

The "Header + Image Model" mode runs both classifiers over every series and has
to reduce their two probability vectors to one before the study-level label
assignment in :func:`mrid_app.processing.optimize_study_selections` runs.

Historically that reduction was a fixed 50/50 arithmetic mean.  That is a sound
default when the two models are of comparable quality, but it degrades the
result whenever one model is substantially weaker than the other on a given
dataset: the weak model's errors are averaged into every prediction with equal
weight.  The rules below let a project pick a reduction that suits its data.

Available methods (``FUSION_METHOD`` in ``config.py``)
-----------------------------------------------------

``average`` (default)
    Arithmetic mean, ``p = (p_rf + p_cnn) / 2``.  The *sum rule* of Kittler et
    al.  Robust and well behaved when both models are similarly accurate; it is
    the historical behavior and remains the default.

``product``
    Normalized elementwise product, ``p ∝ p_rf · p_cnn``.  The *product rule*
    (equivalently a log-opinion pool).  Sharper than the mean and effectively
    lets either model veto a class it considers impossible, which helps when the
    two models make largely independent errors.  Correspondingly less forgiving
    when one model is confidently wrong.

``max``
    Per-class maximum, ``p ∝ max(p_rf, p_cnn)``.  The *max rule*.  The class
    with the single highest posterior across both models wins, so a confident
    model is never diluted by an uncertain one.

``confidence``
    Per-series confidence weighting.  Each model's weight is its normalized
    certainty ``1 - H(p) / log K`` (``H`` = Shannon entropy, ``K`` = number of
    classes), so for each individual series the model that is more certain
    contributes more.  Training-free and adapts series by series rather than
    globally.

``reliability``
    Per-class reliability weighting.  Each model's weight for class ``c`` is the
    F1 score it achieved for that class on its own held-out validation split,
    recorded at training time (see :func:`record_validation_f1`).  This is the
    rule to use when one model is globally much stronger than the other, or when
    each model is stronger on a different subset of classes.  Falls back to
    ``average`` when the recorded validation metrics are unavailable.

All rules return a vector that sums to 1 over the classes, so
``OVERRIDE_THRESHOLD`` keeps the same meaning regardless of the method chosen.
"""

import os

import numpy as np

from .logging_config import get_logger

logger = get_logger()

#: Fusion methods accepted by :func:`fuse_probabilities`, in display order.
FUSION_METHODS = ("average", "product", "max", "confidence", "reliability")

#: Short human-readable descriptions, used by the config wizard tooltips.
FUSION_METHOD_DESCRIPTIONS = {
    "average":     "Mean of both models (sum rule) — the robust default.",
    "product":     "Normalized product (log-opinion pool) — sharper; either model can veto a class.",
    "max":         "Per-class maximum (max rule) — the most confident model wins, never diluted.",
    "confidence":  "Weight each model per series by its certainty (entropy) — training-free, adaptive.",
    "reliability": "Weight each model per class by its validation F1 — best when one model is much weaker.",
}

#: File written next to the models holding each model's per-class validation F1.
RELIABILITY_FILENAME = "fusion_reliability.joblib"

#: Cache for :func:`load_reliability`, keyed on (path, mtime_ns, size).
_reliability_cache = {}

_DEFAULT_METHOD = "average"


def get_configured_method():
    """Return ``FUSION_METHOD`` from the (possibly runtime-loaded) config.

    Unknown or missing values fall back to ``average`` so an out-of-date or
    hand-edited config can never break classification.
    """
    try:
        from . import config as config_module
        method = getattr(config_module, "FUSION_METHOD", _DEFAULT_METHOD)
    except Exception:
        return _DEFAULT_METHOD
    method = str(method or _DEFAULT_METHOD).strip().lower()
    if method not in FUSION_METHODS:
        logger.warning(
            f"Unknown FUSION_METHOD {method!r}; falling back to {_DEFAULT_METHOD!r}. "
            f"Valid options: {', '.join(FUSION_METHODS)}"
        )
        return _DEFAULT_METHOD
    return method


# ── Reliability weights (per-class validation F1) ───────────────────────────

def reliability_path(model_files_dir):
    """Path to the reliability artifact inside *model_files_dir*."""
    return os.path.join(model_files_dir, RELIABILITY_FILENAME)


def record_validation_f1(model_files_dir, model_key, classes, f1_by_class):
    """Record one model's per-class validation F1 for ``reliability`` fusion.

    Called by the header and image trainers at the end of training.  Each
    trainer writes only its own key, so the two can be trained independently and
    in either order; the artifact accumulates whichever models exist.

    Args:
        model_files_dir: the project's ``model_files`` directory
        model_key: ``'rf'`` or ``'cnn'``
        classes: sequence of class names, in label-encoder order
        f1_by_class: sequence of per-class F1 scores aligned with *classes*

    Failures are logged and swallowed — a missing reliability file only causes
    ``reliability`` fusion to fall back to ``average``, never a training failure.
    """
    try:
        import joblib

        path = reliability_path(model_files_dir)
        try:
            data = joblib.load(path) if os.path.exists(path) else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}

        data[model_key] = {
            str(cls): float(score)
            for cls, score in zip(classes, f1_by_class)
            if np.isfinite(score)
        }
        os.makedirs(model_files_dir, exist_ok=True)
        joblib.dump(data, path)
        _reliability_cache.clear()   # training the second model must not read a stale entry
        logger.info(f"Recorded {model_key} per-class validation F1 for fusion: {path}")
    except Exception as e:
        logger.warning(f"Could not record validation F1 for fusion weighting: {e}")


def load_reliability(model_files_dir):
    """Load the reliability artifact, or ``None`` when it is absent/unreadable.

    Cached on (path, mtime) because this is consulted once per study during
    classification and the artifact is a few hundred bytes that rarely change.
    """
    try:
        path = reliability_path(model_files_dir)
        if not os.path.exists(path):
            return None
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        if key in _reliability_cache:
            return _reliability_cache[key]

        import joblib

        data = joblib.load(path)
        data = data if isinstance(data, dict) and data else None
        _reliability_cache.clear()   # only ever one project loaded at a time
        _reliability_cache[key] = data
        return data
    except Exception as e:
        logger.debug(f"Could not load fusion reliability weights: {e}")
        return None


def reliability_weights(reliability, class_names):
    """Build ``(w_rf, w_cnn)`` weight vectors aligned with *class_names*.

    Returns ``None`` when the artifact lacks usable entries for both models, so
    the caller can fall back to ``average``.  Classes missing from the recorded
    metrics fall back to an equal 0.5/0.5 split for that class alone.
    """
    if not reliability:
        return None
    rf_f1 = reliability.get("rf") or {}
    cnn_f1 = reliability.get("cnn") or {}
    if not rf_f1 or not cnn_f1:
        return None

    w_rf = np.empty(len(class_names), dtype=np.float64)
    w_cnn = np.empty(len(class_names), dtype=np.float64)
    for i, cls in enumerate(class_names):
        r = rf_f1.get(str(cls))
        c = cnn_f1.get(str(cls))
        if r is None or c is None or (r + c) <= 0:
            # No usable evidence for this class — weight the two models equally.
            w_rf[i] = w_cnn[i] = 0.5
        else:
            total = float(r) + float(c)
            w_rf[i] = float(r) / total
            w_cnn[i] = float(c) / total
    return w_rf, w_cnn


# ── Fusion rules ────────────────────────────────────────────────────────────

def _normalize(vec):
    """Scale *vec* to sum to 1; return ``None`` if that is not possible."""
    total = float(np.sum(vec))
    if not np.isfinite(total) or total <= 0:
        return None
    return vec / total


def _certainty(p):
    """Normalized certainty ``1 - H(p)/log K`` in [0, 1] (0 = uniform)."""
    k = len(p)
    if k < 2:
        return 1.0
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    entropy = float(-np.sum(q * np.log(q)))
    return float(np.clip(1.0 - entropy / np.log(k), 0.0, 1.0))


def fuse_probabilities(rf_proba, cnn_proba, method=None, weights=None):
    """Reduce header and image probability vectors to a single vector.

    Args:
        rf_proba: header-model probabilities, or ``None``
        cnn_proba: image-model probabilities, or ``None``
        method: one of :data:`FUSION_METHODS`; defaults to ``FUSION_METHOD``
            from the config
        weights: ``(w_rf, w_cnn)`` per-class weight vectors for the
            ``reliability`` method, as returned by :func:`reliability_weights`

    When only one model produced a prediction its vector is returned unchanged,
    which is what every rule degenerates to anyway.  Any rule that cannot
    produce a valid distribution for a particular series (e.g. a product with no
    overlapping support) falls back to the arithmetic mean for that series, so
    fusion never yields an unusable vector.
    """
    if rf_proba is None and cnn_proba is None:
        return None
    if cnn_proba is None:
        return rf_proba
    if rf_proba is None:
        return cnn_proba

    rf = np.asarray(rf_proba, dtype=np.float64)
    cnn = np.asarray(cnn_proba, dtype=np.float64)
    if rf.shape != cnn.shape:
        logger.warning(
            f"Header ({rf.shape}) and image ({cnn.shape}) probability shapes differ; "
            f"using the header model alone for this series."
        )
        return rf_proba

    # The historical default. Both inputs already sum to 1, so the mean does
    # too — computed directly (not via _normalize) to keep it bit-for-bit
    # identical to the behavior that predates the other rules.
    average = (rf + cnn) / 2.0

    if method is None:
        method = get_configured_method()

    if method == "average":
        return average

    if method == "product":
        fused = _normalize(rf * cnn)
        return fused if fused is not None else average

    if method == "max":
        fused = _normalize(np.maximum(rf, cnn))
        return fused if fused is not None else average

    if method == "confidence":
        w_rf = _certainty(rf)
        w_cnn = _certainty(cnn)
        total = w_rf + w_cnn
        if total <= 0:
            # Both models are maximally uncertain — nothing to prefer.
            return average
        return (w_rf * rf + w_cnn * cnn) / total

    if method == "reliability":
        if weights is None:
            return average
        w_rf, w_cnn = weights
        if len(w_rf) != len(rf):
            logger.debug("Reliability weight length does not match the class count; using average.")
            return average
        fused = _normalize(w_rf * rf + w_cnn * cnn)
        return fused if fused is not None else average

    return average
