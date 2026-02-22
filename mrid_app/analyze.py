"""
analyze.py – Unsupervised series-type discovery for the MRID Analyze mode.

Pipeline
--------
1. Load all thumbnails from <output_dir>/series_thumbnails/.
2. Extract a 2048-D feature vector per image using a **frozen ResNet50**
   pretrained on ImageNet (weights already cached locally).  The final FC
   layer is replaced with an Identity so the global-average-pool output
   (2048-D) is used directly.  Thumbnails are converted L→RGB, resized to
   224×224, and normalised with ImageNet statistics.  All inference runs on
   GPU (CUDA) when available, in batches of 64 for throughput.

   Rationale: deep CNN features capture high-level visual semantics
   (intensity distribution, texture, contrast mechanism, anatomy) far more
   discriminatively than hand-crafted pixel / spectral features, and
   transfer well from natural images to medical imaging without any
   fine-tuning.

3. Apply StandardScaler + PCA (up to 150 components) on the 2048-D vectors
   to remove correlated dimensions and speed up subsequent KMeans.

4. Build a *consensus co-occurrence matrix* by running KMeans
   _CONSENSUS_RUNS times with different random seeds.  Entry C[i,j] =
   fraction of runs in which samples i and j were assigned to the same
   cluster.  This averages out initialisation noise.

5. Search K = 2 … K_max using AgglomerativeClustering (average linkage)
   on D = 1 − C.  Accept K if every cluster satisfies:
       • coverage  – fraction of exams with ≥1 series in cluster ≥ threshold
       • min size  – cluster has ≥ _MIN_CLUSTER_FRAC of total samples
   Select the highest K that passes.

6. Return the 4 thumbnails per cluster closest to the cluster centroid.
"""

import os
import logging
import math

import numpy as np
from PIL import Image

logger = logging.getLogger("mrid")

# ── ResNet feature extraction constants ───────────────────────────────────────
_BATCH_SIZE   = 64    # images per GPU batch
_RESNET_INPUT = 224   # ResNet50 canonical input size
_FEATURE_DIM  = 2048  # ResNet50 GAP output dimension

# ── PCA constants ─────────────────────────────────────────────────────────────
_PCA_COMPONENTS = 50

# ── Clustering constants ──────────────────────────────────────────────────────
_MIN_K            = 2
_CONSENSUS_RUNS   = 10
_KMEANS_N_INIT    = 10
_KMEANS_MAX_ITER  = 300
_MIN_CLUSTER_FRAC = 0.02

# ── Misc ──────────────────────────────────────────────────────────────────────
_BEST_EXAMPLES = 4


# ── Filename → exam-id helper ─────────────────────────────────────────────────

def _thumbnail_filename_to_exam_id(filename: str) -> str:
    """
    Derive an exam-level identifier from a thumbnail filename.

    Thumbnail filenames follow the pattern::

        {mrn}_{study_uid}_{series_uid}[_{split_idx:02d}]{ext}

    We use ``{mrn}_{study_uid}`` as the exam key so that different series
    (and split sub-series) from the same study are grouped together.
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


# ── ResNet50 feature extractor ────────────────────────────────────────────────

def _load_resnet_extractor():
    """
    Load a frozen ResNet50 (ImageNet weights) with the FC head replaced by
    Identity so forward() returns the 2048-D GAP vector.

    Returns (model, transform, device).  Falls back to CPU if CUDA is
    unavailable.  Weights are loaded from the local torch hub cache
    (~/.cache/torch/hub/checkpoints/) — no internet access required.
    """
    import torch
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading ResNet50 feature extractor on {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ImageNet normalisation; L images are replicated to 3 channels via
    # convert("RGB") before this transform is applied.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(_RESNET_INPUT),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return model, transform, device


# ── Batch thumbnail loader + feature extractor ────────────────────────────────

def load_all_thumbnail_features(
    thumbnail_dir: str,
    progress_callback=None,
    num_workers: int = 8,
):
    """
    Load every thumbnail in *thumbnail_dir*, run it through the frozen
    ResNet50 feature extractor in GPU batches, and return parallel arrays.

    *num_workers* is accepted for API compatibility but is no longer used
    for CPU thread workers — batched GPU inference is faster.

    *progress_callback(current, total)* is called after each batch so the
    GUI progress bar advances smoothly.

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
        tensors = []
        valid_paths = []

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
                batch_feats = model(batch_tensor).cpu().numpy()  # (B, 2048)
            for i, p in enumerate(valid_paths):
                paths.append(p)
                exam_ids.append(
                    _thumbnail_filename_to_exam_id(os.path.basename(p))
                )
                features.append(batch_feats[i])

        completed += len(batch_paths)
        if progress_callback:
            progress_callback(completed, total)

    if not features:
        return [], [], np.empty((0, _FEATURE_DIM), dtype=np.float32)

    return paths, exam_ids, np.array(features, dtype=np.float32)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_features(features: np.ndarray) -> np.ndarray:
    """
    StandardScale + PCA on the 2048-D ResNet feature matrix.

    Returns the (scaled but not PCA-reduced) feature matrix if PCA cannot
    be applied.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = features.shape
    if n_samples < 2:
        return features

    n_components = min(_PCA_COMPONENTS, n_samples - 1, n_features)
    if n_components < 1:
        return features

    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(scaled)
        logger.debug(
            f"PCA: {n_features}-D → {n_components}-D  "
            f"(variance explained: {pca.explained_variance_ratio_.sum():.1%})"
        )
        return reduced
    except Exception as e:
        logger.warning(f"PCA failed ({e}), using scaled features")
        try:
            return StandardScaler().fit_transform(features)
        except Exception:
            return features


# ── Consensus clustering ──────────────────────────────────────────────────────

def _build_cooccurrence(
    features: np.ndarray,
    k: int,
    n_runs: int,
    n_init: int,
    max_iter: int,
    run_cb=None,
) -> np.ndarray:
    """
    Run KMeans *n_runs* times with different seeds and build a co-occurrence
    matrix C where C[i,j] = fraction of runs in which i and j were in the
    same cluster.

    *run_cb*, if supplied, is called with no arguments after each KMeans run
    completes so callers can track fine-grained progress.

    Returns C as a float32 array of shape (N, N).
    """
    from sklearn.cluster import KMeans

    n = features.shape[0]
    cooc = np.zeros((n, n), dtype=np.float32)

    for seed in range(n_runs):
        try:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                random_state=seed,
                n_init=n_init,
                max_iter=max_iter,
            )
            labels = km.fit_predict(features)
        except Exception as e:
            logger.debug(f"KMeans seed {seed} failed: {e}")
            if run_cb:
                run_cb()
            continue

        # Vectorised co-occurrence update
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) > 1:
                cooc[np.ix_(idx, idx)] += 1.0

        if run_cb:
            run_cb()

    cooc /= max(n_runs, 1)
    np.fill_diagonal(cooc, 1.0)  # each sample is always co-clustered with itself
    return cooc


# ── Coverage and size helpers ─────────────────────────────────────────────────

def _coverage_fraction(labels: np.ndarray, exam_ids: list, k: int) -> list:
    """
    For each cluster c return the fraction of unique exams that contain ≥1
    series assigned to that cluster.
    """
    exam_arr = np.array(exam_ids)
    unique_exams = set(exam_ids)
    n_exams = len(unique_exams)
    if n_exams == 0:
        return [0.0] * k
    fracs = []
    for c in range(k):
        mask = labels == c
        fracs.append(len(set(exam_arr[mask].tolist())) / n_exams)
    return fracs


def _compute_centroids(
    features: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    """Compute per-cluster mean feature vectors."""
    d = features.shape[1]
    centroids = np.zeros((k, d), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(axis=0)
    return centroids


# ── Main cluster-count search ─────────────────────────────────────────────────

def find_optimal_clusters(
    features_reduced: np.ndarray,
    exam_ids: list,
    coverage_threshold: float,
    k_max: int,
    progress_callback=None,
):
    """
    Search K = _MIN_K … k_max using consensus clustering and return the
    highest K where every cluster satisfies the coverage and minimum-size
    criteria.

    Strategy per candidate K
    ~~~~~~~~~~~~~~~~~~~~~~~~
    1.  Build a co-occurrence matrix C over _CONSENSUS_RUNS KMeans runs,
        each run using exactly K clusters.  This ensures the co-occurrence
        signal is tuned to the candidate K, not diluted by running with
        an overly large k_max.
    2.  Cluster the distance matrix D = 1 − C with AgglomerativeClustering
        (average linkage) to get stable, consensus-based labels.
    3.  Evaluate coverage and minimum-size constraints.

    Returns
    -------
    best_k       : int
    best_labels  : np.ndarray  shape (N,)
    best_centers : np.ndarray  shape (K, D)
    """
    from sklearn.cluster import AgglomerativeClustering

    n_samples = features_reduced.shape[0]
    k_max = max(_MIN_K, min(k_max, n_samples - 1))
    min_cluster_size = max(1, int(n_samples * _MIN_CLUSTER_FRAC))

    best_k = 1
    best_labels = np.zeros(n_samples, dtype=int)
    best_centers = features_reduced.mean(axis=0, keepdims=True)

    # Total KMeans runs across all candidate k values — used as the progress
    # denominator so the bar moves continuously from 0 → total without resets.
    k_range = k_max - _MIN_K + 1
    total_runs = k_range * _CONSENSUS_RUNS
    completed_runs = [0]

    def _run_cb():
        completed_runs[0] += 1
        if progress_callback:
            progress_callback(completed_runs[0], total_runs)

    for k in range(_MIN_K, k_max + 1):
        # Build co-occurrence matrix tuned exactly to this k
        logger.debug(f"Building co-occurrence matrix for k={k} ({_CONSENSUS_RUNS} runs)…")
        cooc = _build_cooccurrence(
            features_reduced,
            k=k,
            n_runs=_CONSENSUS_RUNS,
            n_init=_KMEANS_N_INIT,
            max_iter=_KMEANS_MAX_ITER,
            run_cb=_run_cb,
        )
        dist = np.clip((1.0 - cooc).astype(np.float64), 0.0, None)

        try:
            agg = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
            labels = agg.fit_predict(dist)
        except Exception as e:
            logger.warning(f"Agglomerative clustering failed for k={k}: {e}")
            break

        cov_fracs   = _coverage_fraction(labels, exam_ids, k)
        size_counts = [int(np.sum(labels == c)) for c in range(k)]

        cov_ok  = all(f >= coverage_threshold for f in cov_fracs)
        size_ok = all(sc >= min_cluster_size for sc in size_counts)

        logger.debug(
            f"k={k}  coverage={[f'{f:.2f}' for f in cov_fracs]}  "
            f"sizes={size_counts}  cov_ok={cov_ok}  size_ok={size_ok}"
        )

        if cov_ok and size_ok:
            best_k      = k
            best_labels = labels
            best_centers = _compute_centroids(features_reduced, labels, k)
        else:
            logger.info(
                f"Criterion failed at k={k} "
                f"(cov_ok={cov_ok}, size_ok={size_ok}, "
                f"threshold={coverage_threshold:.2f}). "
                f"Optimal k={best_k}."
            )
            break

    return best_k, best_labels, best_centers


# ── Best-example selector ─────────────────────────────────────────────────────

def get_best_examples(
    paths: list,
    features_reduced: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    n_best: int = _BEST_EXAMPLES,
) -> dict:
    """
    For each cluster return up to *n_best* thumbnail paths whose feature
    vectors are closest to the cluster centroid.
    """
    k = centers.shape[0]
    result = {}
    paths_arr = np.array(paths)
    for c in range(k):
        mask    = labels == c
        indices = np.where(mask)[0]
        if len(indices) == 0:
            result[c] = []
            continue
        dists   = np.linalg.norm(features_reduced[indices] - centers[c], axis=1)
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

    Parameters
    ----------
    thumbnail_dir       : str
    hierarchy           : dict  {mrn: {study_uid: {"series": {…}}}}
    coverage_threshold  : float
    num_workers         : int   (accepted for API compat; GPU batching is used)
    load_progress_cb    : callable(current, total) | None
    cluster_progress_cb : callable(current, total) | None

    Returns
    -------
    dict with keys:
        "k"            : int
        "labels"       : np.ndarray
        "centers"      : np.ndarray
        "paths"        : list[str]
        "exam_ids"     : list[str]
        "best_examples": dict  cluster_index → list[str]
        "features"     : np.ndarray  (PCA-reduced)
    """
    # ── 1. ResNet50 feature extraction ────────────────────────────────────────
    logger.info(f"Extracting ResNet50 features from: {thumbnail_dir}")
    paths, exam_ids, features_raw = load_all_thumbnail_features(
        thumbnail_dir,
        progress_callback=load_progress_cb,
        num_workers=num_workers,
    )
    n_loaded = len(paths)
    logger.info(
        f"Extracted features for {n_loaded} thumbnails, "
        f"dim={features_raw.shape[1] if n_loaded else 0}"
    )

    if n_loaded < _MIN_K:
        raise ValueError(
            f"Not enough thumbnails to cluster "
            f"(need at least {_MIN_K}, got {n_loaded})."
        )

    # ── 2. PCA reduction ──────────────────────────────────────────────────────
    logger.info("Reducing 2048-D ResNet features with PCA…")
    features = reduce_features(features_raw)

    # ── 3. Compute K_max from hierarchy ──────────────────────────────────────
    series_counts = [
        len(study_data.get("series", {}))
        for studies in hierarchy.values()
        for study_data in studies.values()
    ]
    avg_series = (
        sum(series_counts) / len(series_counts)
    ) if series_counts else max(4, n_loaded)
    k_max = max(_MIN_K, min(math.floor(avg_series + 4), 30))
    logger.info(
        f"k_max={k_max}  (avg_series/exam={avg_series:.1f}, "
        f"coverage_threshold={coverage_threshold:.2f})"
    )

    # ── 4. Consensus cluster search ───────────────────────────────────────────
    logger.info("Searching for optimal cluster count via consensus clustering…")
    best_k, labels, centers = find_optimal_clusters(
        features, exam_ids, coverage_threshold, k_max,
        progress_callback=cluster_progress_cb,
    )
    logger.info(f"Optimal cluster count: k={best_k}")

    # ── 5. Best examples per cluster ─────────────────────────────────────────
    best_examples = get_best_examples(paths, features, labels, centers)

    return {
        "k":             best_k,
        "labels":        labels,
        "centers":       centers,
        "paths":         paths,
        "exam_ids":      exam_ids,
        "best_examples": best_examples,
        "features":      features,
    }
