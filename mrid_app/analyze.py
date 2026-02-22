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
5. HDBSCAN clustering — min_cluster_size is derived directly from the
   user-specified coverage_threshold:
       min_cluster_size = max(2, ceil(coverage_threshold × n_unique_exams))
   Noise points (label -1) are reassigned to the nearest cluster centroid
   by Euclidean distance in UMAP space so that every thumbnail is labelled.
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
_HDBSCAN_MIN_SAMPLES         = 1    # allow single-linkage at leaf level
_HDBSCAN_CLUSTER_SELECTION   = "eom"  # excess-of-mass: stable, non-over-split

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
    L2-normalise (already done by caller) then PCA to _PCA_VARIANCE_TARGET.

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
    Cluster with HDBSCAN.  min_cluster_size = ceil(coverage_threshold × n_exams)
    so that every returned cluster is present in at least *coverage_threshold*
    fraction of exams.

    Noise points (label -1) are reassigned to the nearest cluster centroid
    in UMAP space so no thumbnail is left unlabelled.

    Returns (labels, n_clusters).
    """
    import hdbscan as _hdbscan

    n_exams = len(set(exam_ids))
    min_cluster_size = max(2, math.ceil(coverage_threshold * n_exams))
    logger.info(
        f"HDBSCAN: coverage_threshold={coverage_threshold:.2f}, "
        f"n_exams={n_exams}, min_cluster_size={min_cluster_size}"
    )

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=_HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=_HDBSCAN_CLUSTER_SELECTION,
    )
    labels = clusterer.fit_predict(features_umap)

    unique_clusters = sorted(set(labels) - {-1})
    n_clusters = len(unique_clusters)
    noise_count = int(np.sum(labels == -1))
    logger.info(f"HDBSCAN: {n_clusters} clusters, {noise_count} noise points")

    if n_clusters == 0:
        # Entire dataset is noise — treat all as one cluster
        logger.warning("HDBSCAN produced no clusters; assigning all to cluster 0")
        labels = np.zeros(len(labels), dtype=int)
        return labels, 1

    # Reassign noise points to the nearest cluster centroid (UMAP space)
    if noise_count > 0:
        # Remap cluster ids to 0…n_clusters-1
        id_map = {old: new for new, old in enumerate(unique_clusters)}
        remapped = np.array([id_map.get(l, -1) for l in labels], dtype=int)

        centroids = np.array([
            features_umap[remapped == c].mean(axis=0)
            for c in range(n_clusters)
        ])
        noise_mask = remapped == -1
        noise_feats = features_umap[noise_mask]
        dists = np.linalg.norm(
            noise_feats[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )  # (n_noise, n_clusters)
        remapped[noise_mask] = np.argmin(dists, axis=1)
        labels = remapped
    else:
        id_map = {old: new for new, old in enumerate(unique_clusters)}
        labels = np.array([id_map[l] for l in labels], dtype=int)

    return labels, n_clusters


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
    labels, k = _hdbscan_cluster(
        features_umap, features_pca, exam_ids, coverage_threshold
    )
    logger.info(f"Optimal cluster count: k={k}")

    # ── 6. Centroids and best examples (PCA space) ────────────────────────────
    centers = _compute_centroids(features_pca, labels, k)
    best_examples = get_best_examples(paths, features_pca, labels, centers)

    return {
        "k":             k,
        "labels":        labels,
        "centers":       centers,
        "paths":         paths,
        "exam_ids":      exam_ids,
        "best_examples": best_examples,
        "features":      features_pca,
    }
