"""
analyze.py – Unsupervised series-type discovery for the MRID Analyze mode.

Pipeline
--------
1. ResNet50 feature extraction (frozen, ImageNet weights, 2048-D GAP output).
2. L2 normalisation of the feature vectors.
3. PCA — reduce dimensionality while retaining ≥ 90 % of variance
   (capped at 150 components and n_samples-1).
4. UMAP — further non-linear reduction to preserve global cluster structure;
   uses n_neighbors=30 and min_dist=0.1 so global topology is kept intact
   (as opposed to min_dist=0.0 which optimises for tight micro-clusters).
5. HDBSCAN clustering — followed by a coverage gate that enforces the
   user-specified coverage_threshold: a cluster is kept only if it appears in
   at least ceil(coverage_threshold × n_unique_exams) *distinct exams*.
   Clusters failing the gate, and HDBSCAN noise points, are reassigned to the
   nearest surviving cluster centroid (Euclidean distance in UMAP space) so
   every thumbnail is labelled.
6. Return the 4 thumbnails per cluster closest to the cluster centroid
   (distances measured in PCA space).
"""

import os
import logging
import math

import numpy as np
from PIL import Image

logger = logging.getLogger("mrid")

# ── ResNet feature extraction ─────────────────────────────────────────────────
_BATCH_SIZE   = 64
_RESNET_INPUT = 224
_FEATURE_DIM  = 2048

# ── PCA ───────────────────────────────────────────────────────────────────────
_PCA_VARIANCE_TARGET = 0.90   # retain at least this fraction of variance
_PCA_MAX_COMPONENTS  = 150    # hard upper bound

# ── UMAP ─────────────────────────────────────────────────────────────────────
_UMAP_N_NEIGHBORS = 30    # large neighbourhood → global structure preserved
_UMAP_MIN_DIST    = 0.1   # moderate spread → global topology intact
_UMAP_METRIC      = "cosine"

# ── HDBSCAN ──────────────────────────────────────────────────────────────────
_HDBSCAN_MIN_SAMPLES         = None    # lower = group everything, higher = more stringent inclusion in clusters, None = defaults to min cluster size
_HDBSCAN_CLUSTER_SELECTION   = "leaf"  # eom = fewer clusters, leaf = more clusters

# ── Misc ──────────────────────────────────────────────────────────────────────
_BEST_EXAMPLES = 4

# Keep for gui.py compatibility (it reads _MIN_K and _CONSENSUS_RUNS to size
# the progress bar).  With HDBSCAN we report 3 progress steps so we expose
# _CONSENSUS_RUNS=3 and _MIN_K=2 as dummies.
_MIN_K          = 2
_CONSENSUS_RUNS = 3


# ── Filename → exam-id helper ─────────────────────────────────────────────────

def _thumbnail_filename_to_exam_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


# ── ResNet50 feature extractor ────────────────────────────────────────────────

def _load_resnet_extractor():
    import torch
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading ResNet50 feature extractor on {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(_RESNET_INPUT),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, device


# ── Batch feature extraction ──────────────────────────────────────────────────

def load_all_thumbnail_features(
    thumbnail_dir: str,
    progress_callback=None,
    num_workers: int = 8,
):
    """
    Extract ResNet50 features for every thumbnail in *thumbnail_dir*.

    Returns
    -------
    paths    : list[str]
    exam_ids : list[str]
    features : np.ndarray  shape (N, 2048)
    """
    import torch

    valid_exts = {".jpg", ".jpeg", ".png"}
    all_files = sorted([
        os.path.join(thumbnail_dir, f)
        for f in os.listdir(thumbnail_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    total = len(all_files)
    if total == 0:
        return [], [], np.empty((0, _FEATURE_DIM), dtype=np.float32)

    model, transform, device = _load_resnet_extractor()
    paths, exam_ids, features = [], [], []
    completed = 0

    for batch_start in range(0, total, _BATCH_SIZE):
        batch_paths = all_files[batch_start: batch_start + _BATCH_SIZE]
        tensors, valid_paths = [], []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
                valid_paths.append(p)
            except Exception as e:
                logger.debug(f"Failed to load {p}: {e}")

        if tensors:
            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                batch_feats = model(batch_tensor).cpu().numpy()
            for i, p in enumerate(valid_paths):
                paths.append(p)
                exam_ids.append(_thumbnail_filename_to_exam_id(os.path.basename(p)))
                features.append(batch_feats[i])

        completed += len(batch_paths)
        if progress_callback:
            progress_callback(completed, total)

    if not features:
        return [], [], np.empty((0, _FEATURE_DIM), dtype=np.float32)

    return paths, exam_ids, np.array(features, dtype=np.float32)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def _pca_to_variance(features_l2: np.ndarray) -> np.ndarray:
    """
    PCA-reduce the (already L2-normalised) features so that at least
    _PCA_VARIANCE_TARGET of the variance is retained.

    Returns the PCA-reduced array.
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = features_l2.shape
    if n_samples < 2:
        return features_l2

    # Fit full PCA first to find the elbow, then refit with exact n_components.
    # Cap at _PCA_MAX_COMPONENTS and n_samples-1.
    max_comp = min(_PCA_MAX_COMPONENTS, n_samples - 1, n_features)
    try:
        pca_full = PCA(n_components=max_comp, random_state=42).fit(features_l2)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, _PCA_VARIANCE_TARGET)) + 1
        n_comp = max(1, min(n_comp, max_comp))
        features_pca = PCA(n_components=n_comp, random_state=42).fit_transform(features_l2)
        logger.info(
            f"PCA: {n_features}-D → {n_comp}-D  "
            f"(variance retained: {cumvar[n_comp - 1]:.1%})"
        )
        return features_pca
    except Exception as e:
        logger.warning(f"PCA failed ({e}), using L2-normalised features")
        return features_l2


def _umap_reduce(features_pca: np.ndarray) -> np.ndarray:
    """
    UMAP reduction for global structure preservation.

    n_components is set to min(n_pca_dims, 15) so we never expand dimensions.
    Returns UMAP embedding, or the PCA features if UMAP fails.
    """
    n_samples, n_pca = features_pca.shape
    n_components = max(2, min(n_pca, 15))
    n_neighbors  = min(_UMAP_N_NEIGHBORS, n_samples - 1)

    try:
        import os as _os
        _os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
        import umap as _umap
        reducer = _umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=_UMAP_MIN_DIST,
            metric=_UMAP_METRIC,
            random_state=42,
            n_jobs=1,       # explicit — silences "n_jobs overridden" warning
            low_memory=False,
        )
        embedding = reducer.fit_transform(features_pca).astype(np.float32)
        logger.info(f"UMAP: {n_pca}-D → {n_components}-D")
        return embedding
    except Exception as e:
        logger.warning(f"UMAP failed ({e}), using PCA features for clustering")
        return features_pca.astype(np.float32)


# ── HDBSCAN clustering ────────────────────────────────────────────────────────

def _hdbscan_cluster(
    features_umap: np.ndarray,
    features_pca: np.ndarray,
    exam_ids: list,
    coverage_threshold: float,
) -> tuple:
    """
    Cluster with HDBSCAN, then enforce ANALYZE_COVERAGE_THRESHOLD as a coverage
    gate: a cluster is kept only if it appears in at least
    ``ceil(coverage_threshold × n_exams)`` *distinct exams*. This prevents small
    series types — including ones where a single exam contributes many similar
    series — from forming their own cluster.

    Clusters that fail the coverage gate, along with HDBSCAN noise points, are
    reassigned to the nearest *surviving* cluster centroid in UMAP space (with
    membership probability 0.0) so no thumbnail is left unlabelled.

    Returns (labels, probabilities, n_clusters).
    """
    import hdbscan as _hdbscan

    n_exams = len(set(exam_ids))
    # A cluster covering K distinct exams necessarily has >= K points, so this
    # min_cluster_size is a safe lower bound that never drops a cluster that
    # would pass the coverage gate; the distinct-exam gate below is authoritative.
    min_exams = max(1, math.ceil(coverage_threshold * n_exams))
    min_cluster_size = max(2, min_exams)
    logger.info(
        f"HDBSCAN: coverage_threshold={coverage_threshold:.2f}, "
        f"n_exams={n_exams}, min_exams={min_exams}, min_cluster_size={min_cluster_size}"
    )

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples= _HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=_HDBSCAN_CLUSTER_SELECTION,
    )
    labels = clusterer.fit_predict(features_umap)
    probabilities = np.array(clusterer.probabilities_, dtype=np.float32)

    return _apply_coverage_gate(labels, probabilities, features_umap, exam_ids, min_exams)


def _apply_coverage_gate(labels, probabilities, features_umap, exam_ids, min_exams):
    """
    Enforce the distinct-exam coverage gate on raw HDBSCAN output.

    Keeps only clusters present in >= ``min_exams`` distinct exams. Clusters
    failing the gate, and HDBSCAN noise points (label -1), are reassigned to the
    nearest surviving cluster centroid (Euclidean distance in UMAP space) with
    membership probability 0.0. Returns (labels, probabilities, n_clusters).
    """
    labels = np.asarray(labels)
    probabilities = np.array(probabilities, dtype=np.float32)
    exam_arr = np.asarray(exam_ids)
    n_exams = len(set(exam_ids))

    valid_clusters = []
    for c in sorted(set(labels.tolist()) - {-1}):
        n_cov = len(set(exam_arr[labels == c]))
        if n_cov >= min_exams:
            valid_clusters.append(c)
        else:
            logger.info(
                f"HDBSCAN: dropping cluster {c} — covers {n_cov}/{n_exams} exams "
                f"(< required {min_exams})"
            )

    noise_count = int(np.sum(labels == -1))
    logger.info(
        f"HDBSCAN: {len(valid_clusters)} clusters meet coverage, "
        f"{noise_count} noise points"
    )

    if not valid_clusters:
        # No cluster meets the coverage threshold — treat all as one cluster
        logger.warning("No cluster meets the coverage threshold; assigning all to cluster 0")
        return (np.zeros(len(labels), dtype=int),
                np.zeros(len(labels), dtype=np.float32), 1)

    # Remap surviving clusters to 0…n_clusters-1; everything else (noise +
    # dropped clusters) becomes -1 and is reassigned below.
    id_map = {old: new for new, old in enumerate(valid_clusters)}
    remapped = np.array([id_map.get(l, -1) for l in labels], dtype=int)
    n_clusters = len(valid_clusters)

    unassigned_mask = remapped == -1
    if unassigned_mask.any():
        # Centroids from surviving-cluster members only
        centroids = np.array([
            features_umap[remapped == c].mean(axis=0)
            for c in range(n_clusters)
        ])
        feats = features_umap[unassigned_mask]
        dists = np.linalg.norm(
            feats[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )  # (n_unassigned, n_clusters)
        remapped[unassigned_mask] = np.argmin(dists, axis=1)
        # Reassigned points are not confident members of their new cluster.
        probabilities[unassigned_mask] = 0.0

    return remapped, probabilities, n_clusters


# ── Cluster representative name helpers ──────────────────────────────────────

def _parse_thumbnail_stem(stem: str) -> tuple:
    """
    Parse a thumbnail stem of the form ``{mrn}_{study_uid}_{series_uid}[_{split_idx}]``
    into ``(mrn, study_uid, series_uid)``.

    DICOM UIDs contain dots and no underscores, so the last underscore-delimited
    component that is all-digits is the optional split index.  Everything else is
    split as mrn (first token), study_uid (second token), series_uid (remainder).
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, "", ""
    # Strip trailing split index (2-digit numeric suffix)
    if parts[-1].isdigit() and len(parts) > 3:
        parts = parts[:-1]
    mrn = parts[0]
    study_uid = parts[1]
    series_uid = "_".join(parts[2:])
    return mrn, study_uid, series_uid


def _build_thumbnail_to_dicom_map(hierarchy: dict) -> dict:
    """
    Walk the hierarchy and return a dict mapping each thumbnail *basename*
    (without directory) to a DICOM file path suitable for reading with pydicom.

    hierarchy structure:
        hierarchy[mrn][study_uid]["series"][series_desc] = [
            {"dicom_path": ..., "thumbnail_path": ...}, ...
        ]
    """
    mapping = {}
    if not hierarchy:
        return mapping
    for mrn, studies in hierarchy.items():
        if not isinstance(studies, dict):
            continue
        for study_uid, study_data in studies.items():
            if not isinstance(study_data, dict):
                continue
            series_dict = study_data.get("series", {})
            if not isinstance(series_dict, dict):
                continue
            for _desc, entries in series_dict.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    thumb = entry.get("thumbnail_path")
                    dicom = entry.get("dicom_path")
                    if thumb and dicom:
                        mapping[os.path.basename(thumb)] = dicom
    return mapping


def _read_series_description(dicom_path: str) -> str:
    """Read SeriesDescription from a DICOM file; return '' on failure."""
    try:
        import pydicom
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        desc = getattr(ds, "SeriesDescription", None)
        return str(desc).strip() if desc else ""
    except Exception:
        return ""


def _levenshtein_medoid(strings: list) -> str:
    """
    Return the string from *strings* that minimises the total pairwise
    Levenshtein distance to all other strings (i.e. the medoid).

    Falls back to the first string if the list is empty or Levenshtein fails.
    """
    if not strings:
        return ""
    if len(strings) == 1:
        return strings[0]
    try:
        import Levenshtein
        n = len(strings)
        total_dist = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                d = Levenshtein.distance(strings[i], strings[j])
                total_dist[i] += d
                total_dist[j] += d
        return strings[int(np.argmin(total_dist))]
    except Exception:
        return strings[0]


_HIGH_CONF_THRESHOLD = 0.8   # membership probability threshold for representative selection


def get_representative_names(
    paths: list,
    labels: np.ndarray,
    probabilities: np.ndarray,
    k: int,
    hierarchy: dict,
) -> list:
    """
    For each cluster 0…k-1, identify high-confidence members (probability ≥
    _HIGH_CONF_THRESHOLD), read their SeriesDescription from the corresponding
    DICOM file (looked up via *hierarchy*), and return the Levenshtein medoid
    of those descriptions as the cluster's representative name.

    If no high-confidence members exist for a cluster, falls back to all members.
    If no descriptions can be read, returns ``"Series{c+1}"`` as a placeholder.

    Returns a list of k strings.
    """
    thumb_to_dicom = _build_thumbnail_to_dicom_map(hierarchy)
    paths_arr = np.array(paths)
    names = []

    for c in range(k):
        cluster_mask = labels == c
        cluster_indices = np.where(cluster_mask)[0]

        # Prefer high-confidence members; fall back to all members
        high_conf_mask = cluster_mask & (probabilities >= _HIGH_CONF_THRESHOLD)
        hc_indices = np.where(high_conf_mask)[0]
        candidate_indices = hc_indices if len(hc_indices) >= 1 else cluster_indices

        descriptions = []
        for idx in candidate_indices:
            thumb_basename = os.path.basename(paths_arr[idx])
            dicom_path = thumb_to_dicom.get(thumb_basename)
            if dicom_path:
                desc = _read_series_description(dicom_path)
                if desc:
                    descriptions.append(desc)

        if descriptions:
            rep = _levenshtein_medoid(descriptions)
            # Sanitize: strip leading/trailing whitespace, replace characters
            # unsafe for filenames with underscores (keep alphanums, dots, dashes).
            import re
            rep = re.sub(r'[^\w.\- ]', '_', rep).strip()
            rep = rep if rep else f"Series{c + 1}"
        else:
            rep = f"Series{c + 1}"

        logger.info(f"Cluster {c}: representative name = {rep!r} "
                    f"(from {len(descriptions)} descriptions, "
                    f"{len(hc_indices)} high-confidence members)")
        names.append(rep)

    return names


# ── Centroid and best-example helpers ────────────────────────────────────────

def _compute_centroids(features: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    d = features.shape[1]
    centroids = np.zeros((k, d), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(axis=0)
    return centroids


def get_best_examples(
    paths: list,
    features_pca: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    n_best: int = _BEST_EXAMPLES,
) -> dict:
    """
    For each cluster return up to *n_best* paths closest to the centroid
    (distances in PCA space).
    """
    k = centers.shape[0]
    result = {}
    paths_arr = np.array(paths)
    for c in range(k):
        mask = labels == c
        indices = np.where(mask)[0]
        if len(indices) == 0:
            result[c] = []
            continue
        dists = np.linalg.norm(features_pca[indices] - centers[c], axis=1)
        top_idx = indices[np.argsort(dists)[:n_best]]
        result[c] = paths_arr[top_idx].tolist()
    return result


# ── Public entry point ────────────────────────────────────────────────────────

def run_analysis(
    thumbnail_dir: str,
    hierarchy: dict,
    coverage_threshold: float,
    num_workers: int = 8,
    load_progress_cb=None,
    cluster_progress_cb=None,
):
    """
    Full analysis pipeline.

    Progress steps reported via cluster_progress_cb(current, 3),
    called *before* each stage begins:
        0 / 3  — init + PCA starting
        1 / 3  — UMAP starting
        2 / 3  — HDBSCAN starting

    Returns dict with keys:
        "k", "labels", "centers", "paths", "exam_ids",
        "best_examples", "features"  (PCA-reduced)
    """

    # ── 1. ResNet50 feature extraction ───────────────────────────────────────
    logger.info(f"Extracting ResNet50 features from: {thumbnail_dir}")
    paths, exam_ids, features_raw = load_all_thumbnail_features(
        thumbnail_dir,
        progress_callback=load_progress_cb,
        num_workers=num_workers,
    )
    n_loaded = len(paths)
    logger.info(f"Extracted features for {n_loaded} thumbnails")

    if n_loaded < 2:
        raise ValueError(
            f"Not enough thumbnails to cluster (need ≥2, got {n_loaded})."
        )

    # ── 2. L2 normalisation ───────────────────────────────────────────────────
    from sklearn.preprocessing import normalize
    features_l2 = normalize(features_raw, norm="l2")

    # ── 3. PCA to ≥90 % variance ─────────────────────────────────────────────
    # Signal current=0: init the bar AND show "PCA…" before PCA starts.
    if cluster_progress_cb:
        cluster_progress_cb(0, 3)
    logger.info("PCA dimensionality reduction…")
    features_pca = _pca_to_variance(features_l2)

    # ── 4. UMAP ───────────────────────────────────────────────────────────────
    # Signal current=1: show "UMAP…" before UMAP starts.
    if cluster_progress_cb:
        cluster_progress_cb(1, 3)
    logger.info("UMAP embedding…")
    features_umap = _umap_reduce(features_pca)

    # ── 5. HDBSCAN ────────────────────────────────────────────────────────────
    # Signal current=2: show "HDBSCAN…" before clustering starts.
    if cluster_progress_cb:
        cluster_progress_cb(2, 3)
    logger.info("HDBSCAN clustering…")
    labels, probabilities, k = _hdbscan_cluster(
        features_umap, features_pca, exam_ids, coverage_threshold
    )
    logger.info(f"Optimal cluster count: k={k}")

    # ── 6. Centroids and best examples (PCA space) ────────────────────────────
    centers = _compute_centroids(features_pca, labels, k)
    best_examples = get_best_examples(paths, features_pca, labels, centers)

    # ── 7. Representative names from SeriesDescription medoids ───────────────
    logger.info("Deriving cluster representative names from SeriesDescriptions…")
    representative_names = get_representative_names(
        paths, labels, probabilities, k, hierarchy
    )

    return {
        "k":                   k,
        "labels":              labels,
        "probabilities":       probabilities,
        "centers":             centers,
        "paths":               paths,
        "exam_ids":            exam_ids,
        "best_examples":       best_examples,
        "features":            features_pca,
        "representative_names": representative_names,
    }
