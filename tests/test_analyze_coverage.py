"""Tests for the Analyze-mode coverage gate (ANALYZE_COVERAGE_THRESHOLD).

A cluster must appear in at least ceil(coverage_threshold * n_exams) distinct
exams, otherwise it is dropped and its points are merged into the nearest
surviving cluster. These exercise the pure gate logic and need no hdbscan.
"""

import numpy as np

from mrid_app.analyze import _apply_coverage_gate


def test_low_exam_cluster_is_dropped():
    """Two well-separated blobs: one spans many exams, one spans only 2.
    With min_exams=4 the low-coverage blob is dropped and merged."""
    # Blob A near origin, 30 points across 15 exams (2 each)
    a = np.random.RandomState(0).randn(30, 2) * 0.3
    a_exams = [f"A{i // 2}" for i in range(30)]            # 15 distinct exams
    # Blob B far away, 30 points from only 2 exams
    b = np.random.RandomState(1).randn(30, 2) * 0.3 + [50, 50]
    b_exams = [f"B{i // 15}" for i in range(30)]           # 2 distinct exams

    feats = np.vstack([a, b]).astype(np.float32)
    exam_ids = a_exams + b_exams
    raw_labels = np.array([0] * 30 + [1] * 30)             # as HDBSCAN would label
    probs = np.ones(60, dtype=np.float32)

    labels, out_probs, k = _apply_coverage_gate(raw_labels, probs, feats, exam_ids, min_exams=4)

    assert k == 1, f"low-coverage cluster should be dropped, got {k} clusters"
    assert set(labels.tolist()) == {0}
    # The merged-in (formerly cluster 1) points must have probability 0
    assert np.all(out_probs[30:] == 0.0)
    # Surviving cluster's own points keep their probability
    assert np.all(out_probs[:30] == 1.0)


def test_both_clusters_kept_when_coverage_met():
    a = np.random.RandomState(0).randn(20, 2) * 0.3
    a_exams = [f"A{i}" for i in range(20)]                 # 20 distinct exams
    b = np.random.RandomState(1).randn(20, 2) * 0.3 + [50, 50]
    b_exams = [f"B{i}" for i in range(20)]                 # 20 distinct exams
    feats = np.vstack([a, b]).astype(np.float32)
    exam_ids = a_exams + b_exams
    raw_labels = np.array([0] * 20 + [1] * 20)
    probs = np.ones(40, dtype=np.float32)

    labels, _, k = _apply_coverage_gate(raw_labels, probs, feats, exam_ids, min_exams=4)
    assert k == 2
    assert set(labels.tolist()) == {0, 1}


def test_noise_points_reassigned():
    feats = np.vstack([
        np.random.RandomState(0).randn(10, 2) * 0.3,
        np.array([[100.0, 100.0]]),  # a noise point far away
    ]).astype(np.float32)
    exam_ids = [f"A{i}" for i in range(10)] + ["A0"]
    raw_labels = np.array([0] * 10 + [-1])
    probs = np.concatenate([np.ones(10, dtype=np.float32), [0.0]])

    labels, out_probs, k = _apply_coverage_gate(raw_labels, probs, feats, exam_ids, min_exams=3)
    assert k == 1
    assert labels[-1] == 0           # noise reassigned to the only cluster
    assert out_probs[-1] == 0.0      # with probability 0


def test_all_dropped_falls_back_to_single_cluster():
    feats = np.random.RandomState(0).randn(6, 2).astype(np.float32)
    exam_ids = ["A0", "A0", "A0", "B0", "B0", "B0"]   # 2 distinct exams total
    raw_labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.ones(6, dtype=np.float32)

    # min_exams far higher than any cluster's coverage -> fallback to 1 cluster
    labels, _, k = _apply_coverage_gate(raw_labels, probs, feats, exam_ids, min_exams=5)
    assert k == 1
    assert set(labels.tolist()) == {0}
