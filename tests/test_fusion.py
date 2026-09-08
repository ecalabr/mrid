"""Tests for the header/image probability fusion rules (FUSION_METHOD)."""

import os
import tempfile

import numpy as np

from mrid_app import fusion
from mrid_app.fusion import (FUSION_METHODS, fuse_probabilities, get_configured_method,
                             load_reliability, record_validation_f1, reliability_weights)


def _sums_to_one(vec):
    return abs(float(np.sum(vec)) - 1.0) < 1e-9


def test_average_is_bit_for_bit_the_historical_mean():
    """'average' must stay exactly (rf + cnn) / 2 — it is the default, and the
    OVERRIDE_THRESHOLD comparison downstream is knife-edge sensitive."""
    rf = np.array([0.7, 0.2, 0.1])
    cnn = np.array([0.1, 0.6, 0.3])
    fused = fuse_probabilities(rf, cnn, method="average")
    assert np.array_equal(fused, (rf + cnn) / 2)


def test_all_methods_stay_on_the_probability_scale():
    """Every rule must return scores in [0, 1] bounded by what the two models
    themselves said, so OVERRIDE_THRESHOLD keeps its meaning whichever method is
    configured. Only some rules also sum to 1 — see the next test."""
    rf = np.array([0.7, 0.2, 0.1])
    cnn = np.array([0.1, 0.6, 0.3])
    weights = (np.array([0.8, 0.8, 0.8]), np.array([0.2, 0.2, 0.2]))
    lo, hi = np.minimum(rf, cnn), np.maximum(rf, cnn)
    for method in FUSION_METHODS:
        fused = fuse_probabilities(rf, cnn, method=method, weights=weights)
        assert np.all(fused >= 0.0) and np.all(fused <= 1.0), \
            f"{method} left [0, 1]: {fused}"
        if method != "max":   # the max rule is renormalized, so it can exceed hi
            assert np.all(fused <= hi + 1e-9), f"{method} exceeded both models: {fused}"


def test_only_the_normalizing_rules_sum_to_one():
    """'product' is a geometric mean and deliberately does not normalize; the
    others do. This is the property the single-target-series bug turned on."""
    rf = np.array([0.7, 0.2, 0.1])
    cnn = np.array([0.1, 0.6, 0.3])
    for method in ("average", "max", "confidence"):
        assert _sums_to_one(fuse_probabilities(rf, cnn, method=method)), method
    assert not _sums_to_one(fuse_probabilities(rf, cnn, method="product"))


def test_single_model_passes_through_unchanged():
    """With only one model available every rule degenerates to that model."""
    rf = np.array([0.7, 0.2, 0.1])
    for method in FUSION_METHODS:
        assert np.array_equal(fuse_probabilities(rf, None, method=method), rf)
        assert np.array_equal(fuse_probabilities(None, rf, method=method), rf)
    assert fuse_probabilities(None, None, method="average") is None


def test_product_rule_lets_either_model_veto():
    """A class one model rules out cannot win under the product rule, even when
    the other model is confident about it."""
    rf = np.array([0.0, 0.5, 0.5])
    cnn = np.array([0.9, 0.05, 0.05])
    fused = fuse_probabilities(rf, cnn, method="product")
    assert fused[0] == 0.0
    assert int(np.argmax(fused)) != 0
    # The mean, by contrast, still lets the vetoed class win.
    assert int(np.argmax(fuse_probabilities(rf, cnn, method="average"))) == 0


def test_product_scores_zero_when_the_models_completely_disagree():
    """With disjoint support every class is vetoed by one model or the other.
    Scoring it all zero means nothing clears the threshold and the series falls
    through to 'Other' — the honest outcome when the models flatly disagree."""
    rf = np.array([1.0, 0.0])
    cnn = np.array([0.0, 1.0])
    fused = fuse_probabilities(rf, cnn, method="product")
    assert np.allclose(fused, [0.0, 0.0])


def test_product_reflects_header_confidence_with_a_saturated_image_model():
    """Regression for the single-target-series bug: with one target class the
    image model can only ever say 'target', and the old normalized product then
    returned 1.0 for every series regardless of what the header model thought,
    so nothing could be told apart."""
    cnn = np.array([0.0, 1.0])            # [Other, target] — saturated
    weak = fuse_probabilities(np.array([0.90, 0.10]), cnn, method="product")
    strong = fuse_probabilities(np.array([0.15, 0.85]), cnn, method="product")

    assert weak[1] < 0.5, f"a header probability of 0.10 must not select: {weak[1]}"
    assert strong[1] > 0.5, f"a header probability of 0.85 must select: {strong[1]}"
    assert strong[1] > weak[1] + 0.3, "the two must be clearly separated"


def test_max_rule_picks_the_globally_most_confident_class():
    """The class with the single highest posterior across both models wins,
    rather than being diluted by the other model's uncertainty."""
    rf = np.array([0.34, 0.33, 0.33])   # nearly uniform
    cnn = np.array([0.05, 0.90, 0.05])  # confident about class 1
    fused = fuse_probabilities(rf, cnn, method="max")
    assert int(np.argmax(fused)) == 1


def test_confidence_weighting_favors_the_more_certain_model():
    """A near-uniform model should barely move the result."""
    rf = np.array([0.34, 0.33, 0.33])
    cnn = np.array([0.02, 0.96, 0.02])
    fused = fuse_probabilities(rf, cnn, method="confidence")
    assert int(np.argmax(fused)) == 1
    # The confident model dominates, so the result sits closer to it than the
    # plain mean does.
    assert fused[1] > float(((rf + cnn) / 2)[1])


def test_confidence_falls_back_when_both_models_are_uniform():
    """Two maximally uncertain models give no reason to prefer either."""
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    fused = fuse_probabilities(uniform, uniform, method="confidence")
    assert np.allclose(fused, uniform)


def test_reliability_weighting_suppresses_the_weaker_model():
    """This is the tcga-brca failure mode: a weak image model should not be
    able to drag a strong header model off the correct class."""
    rf = np.array([0.6, 0.4])    # strong model, correct (class 0)
    cnn = np.array([0.1, 0.9])   # weak model, confidently wrong
    # Header F1 0.9 vs image F1 0.1 on the held-out validation split.
    weights = reliability_weights(
        {"rf": {"T1": 0.9, "T2": 0.9}, "cnn": {"T1": 0.1, "T2": 0.1}},
        ["T1", "T2"],
    )
    assert int(np.argmax(fuse_probabilities(rf, cnn, method="reliability",
                                            weights=weights))) == 0
    # The plain mean gets this wrong.
    assert int(np.argmax(fuse_probabilities(rf, cnn, method="average"))) == 1


def test_reliability_weights_are_per_class():
    """Each model can be the stronger one for a different class."""
    w_rf, w_cnn = reliability_weights(
        {"rf": {"T1": 0.9, "T2": 0.1}, "cnn": {"T1": 0.1, "T2": 0.9}},
        ["T1", "T2"],
    )
    assert np.allclose(w_rf, [0.9, 0.1])
    assert np.allclose(w_cnn, [0.1, 0.9])


def test_reliability_falls_back_without_usable_metrics():
    """Missing or one-sided metrics fall back to the mean rather than failing."""
    assert reliability_weights(None, ["T1"]) is None
    assert reliability_weights({}, ["T1"]) is None
    assert reliability_weights({"rf": {"T1": 0.9}}, ["T1"]) is None   # no cnn key

    rf, cnn = np.array([0.7, 0.3]), np.array([0.2, 0.8])
    fused = fuse_probabilities(rf, cnn, method="reliability", weights=None)
    assert np.array_equal(fused, (rf + cnn) / 2)


def test_reliability_weights_unknown_class_splits_evenly():
    """A class absent from the recorded metrics weights both models equally."""
    w_rf, w_cnn = reliability_weights(
        {"rf": {"T1": 0.8}, "cnn": {"T1": 0.2}}, ["T1", "FLAIR"]
    )
    assert np.allclose(w_rf, [0.8, 0.5])
    assert np.allclose(w_cnn, [0.2, 0.5])


def test_record_and_load_validation_f1_round_trip():
    """Each trainer writes only its own key, so the two models can be trained
    independently and in either order."""
    with tempfile.TemporaryDirectory() as tmp:
        model_files = os.path.join(tmp, "model_files")
        record_validation_f1(model_files, "rf", ["T1", "T2"], [0.9, 0.8])
        assert load_reliability(model_files) == {"rf": {"T1": 0.9, "T2": 0.8}}

        record_validation_f1(model_files, "cnn", ["T1", "T2"], [0.5, 0.6])
        data = load_reliability(model_files)
        assert data["rf"] == {"T1": 0.9, "T2": 0.8}
        assert data["cnn"] == {"T1": 0.5, "T2": 0.6}

        weights = reliability_weights(data, ["T1", "T2"])
        assert weights is not None


def test_load_reliability_missing_dir_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_reliability(os.path.join(tmp, "nope")) is None


def test_record_validation_f1_skips_non_finite_scores():
    """A class with no validation samples yields NaN; it must not be recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        model_files = os.path.join(tmp, "model_files")
        record_validation_f1(model_files, "rf", ["T1", "T2"], [0.9, float("nan")])
        assert load_reliability(model_files) == {"rf": {"T1": 0.9}}


def test_mismatched_shapes_fall_back_to_the_header_model():
    """A stale model paired with a newer label set must not crash inference."""
    rf = np.array([0.5, 0.5])
    cnn = np.array([0.3, 0.3, 0.4])
    assert np.array_equal(fuse_probabilities(rf, cnn, method="average"), rf)


def test_unknown_configured_method_falls_back_to_average():
    """A hand-edited or out-of-date config can never break classification."""
    from mrid_app import config as config_module

    original = getattr(config_module, "FUSION_METHOD", "average")
    try:
        config_module.FUSION_METHOD = "not-a-real-method"
        assert get_configured_method() == "average"
        config_module.FUSION_METHOD = "RELIABILITY"   # case/whitespace tolerant
        assert get_configured_method() == "reliability"
        del config_module.FUSION_METHOD               # older config with no key
        assert get_configured_method() == "average"
    finally:
        config_module.FUSION_METHOD = original


def test_config_default_is_reliability():
    """New projects get the rule that scored best across the evaluation
    datasets; see the FUSION_METHOD comment in config.py."""
    from mrid_app import config as config_module

    assert config_module.FUSION_METHOD == "reliability"


def test_default_rule_handles_a_saturated_image_model():
    """The default must not repeat the failure that made 'product' unusable on a
    single-target-series project: with one target class the image model can only
    say 'target', and the default must still follow the header model."""
    cnn = np.array([0.0, 1.0])                      # [Other, target], saturated
    weights = reliability_weights(
        {"rf": {"Other": 0.9, "target": 0.9}, "cnn": {"Other": 0.1, "target": 0.1}},
        ["Other", "target"])

    weak = fuse_probabilities(np.array([0.90, 0.10]), cnn,
                              method="reliability", weights=weights)
    strong = fuse_probabilities(np.array([0.15, 0.85]), cnn,
                                method="reliability", weights=weights)
    assert weak[1] < 0.5 < strong[1], f"weak={weak[1]:.3f} strong={strong[1]:.3f}"


def test_missing_config_key_still_falls_back_to_average():
    """A config written before FUSION_METHOD existed must keep the behavior it
    had when it was written — the fallback is 'average', NOT the new default."""
    from mrid_app import config as config_module

    original = getattr(config_module, "FUSION_METHOD", "reliability")
    try:
        del config_module.FUSION_METHOD
        assert get_configured_method() == "average"
    finally:
        config_module.FUSION_METHOD = original


def test_every_method_has_a_wizard_description():
    """The config wizard renders one tooltip per option."""
    assert set(fusion.FUSION_METHOD_DESCRIPTIONS) == set(FUSION_METHODS)


# ── Integration: the rule actually reaches optimize_study_selections ─────────

class _FakeEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)


def _run_optimize(method, rf_vec, cnn_vec, classes, model_files_dir=None):
    """Drive optimize_study_selections with stubbed model inference so the test
    exercises the fusion wiring and label assignment, not DICOM I/O."""
    from mrid_app import processing
    from mrid_app import config as config_module

    study_dict = {"series_a": ["/fake/a.dcm"]}
    encoder = _FakeEncoder(classes)

    orig_rf = processing.get_series_probabilities_batch
    orig_cnn = processing.get_image_probabilities_batch
    orig_3d = processing.check_is_3d_acquisition
    orig_method = getattr(config_module, "FUSION_METHOD", "average")
    try:
        processing.get_series_probabilities_batch = lambda *a, **k: [np.array(rf_vec)]
        processing.get_image_probabilities_batch = lambda *a, **k: [np.array(cnn_vec)]
        processing.check_is_3d_acquisition = lambda *a, **k: False
        config_module.FUSION_METHOD = method
        assignments, _ = processing.optimize_study_selections(
            study_dict, "Header + Image Model",
            "rf-model", encoder, "rf-feature-encoder",
            "cnn-model", encoder, None,
            model_files_dir=model_files_dir,
        )
        return assignments["/fake/a.dcm"]
    finally:
        processing.get_series_probabilities_batch = orig_rf
        processing.get_image_probabilities_batch = orig_cnn
        processing.check_is_3d_acquisition = orig_3d
        config_module.FUSION_METHOD = orig_method


def test_optimize_study_selections_honors_the_configured_method():
    """The same probabilities must produce different labels under different
    rules — proof the config value reaches the fusion step."""
    classes = ["T1", "T2", "FLAIR"]
    rf = [0.30, 0.90, 0.10]    # header: confident T2
    cnn = [0.20, 0.85, 0.30]   # image: agrees on T2

    # Both models are confident about T2, so the geometric mean stays high and
    # the label is assigned. Weighting the image model to nothing while the
    # header disagrees, by contrast, leaves nothing over the threshold.
    assert _run_optimize("product", rf, cnn, classes) == "T2"
    assert _run_optimize("average", rf, cnn, classes) == "T2"

    # Where the rules genuinely differ: the image model is sure, the header
    # model is not. Averaging clears the threshold on the image model's
    # confidence alone (0.55); the product rule requires both to be convinced,
    # and sqrt(0.15 * 0.95) = 0.38 is not enough.
    rf_weak = [0.40, 0.15, 0.10]
    cnn_sure = [0.03, 0.95, 0.02]
    assert _run_optimize("average", rf_weak, cnn_sure, classes) == "T2"
    assert _run_optimize("product", rf_weak, cnn_sure, classes) == "Other"


def test_optimize_study_selections_uses_recorded_reliability_weights():
    """With a reliability artifact on disk the weaker model is down-weighted;
    without one the same call falls back to the mean."""
    classes = ["T1", "T2", "FLAIR"]
    rf = [0.60, 0.30, 0.10]    # header: T1
    cnn = [0.05, 0.60, 0.35]   # image: T2, and it is the unreliable model here

    with tempfile.TemporaryDirectory() as tmp:
        model_files = os.path.join(tmp, "model_files")
        # Header model is strong, image model is near-useless.
        record_validation_f1(model_files, "rf", classes, [0.95, 0.95, 0.95])
        record_validation_f1(model_files, "cnn", classes, [0.05, 0.05, 0.05])
        assert _run_optimize("reliability", rf, cnn, classes, model_files) == "T1"

        # No artifact -> falls back to 'average' (peak 0.45), which clears no
        # threshold, so the disagreement resolves to Other instead.
        empty = os.path.join(tmp, "absent")
        assert _run_optimize("reliability", rf, cnn, classes, empty) == "Other"
