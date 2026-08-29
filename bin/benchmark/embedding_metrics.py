#!/usr/bin/env python3
"""Score cell embeddings: biological conservation and batch mixing.

Biological conservation (label_key): NMI, HOM, COM, FMI, ARI on cluster labels
(Leiden with the resolution matched to the number of cell types, or KMeans with
k = number of cell types), ASW, cLISI, Acc@kNN (5-fold, k=5) and graph
connectivity. Batch mixing (batch_key): kBET, BRAS, iLISI and CiLISI.

Output layout under --output-dir (per clustering choice, <G> = bio_conservation_leiden
or bio_conservation, <B> = batch_mixing_leiden or batch_mixing):
    <G>_metrics/<method>/<sample>_<method>.csv          per-sample long table
    <G>_predictions/<method>/<sample>_<method>_<clustering>_labels.tsv
    <B>_metrics/<method>/<sample>_<method>.csv
    <method>_<G>_metrics_{long,wide}.csv, <method>_<B>_metrics_{long,wide}.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

# These benchmarks are CPU metrics. Set this before importing scib_metrics/JAX
# so JAX does not try to initialize CUDA in CPU-only container runs.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scib_metrics
import sklearn.cluster
import sklearn.metrics
from pynndescent import NNDescent
from scib_metrics.nearest_neighbors import NeighborsResults
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore")

BIO_METRICS = ["NMI", "HOM", "COM", "FMI", "ARI", "ASW", "cLISI", "Acc@kNN", "GC"]
BATCH_METRICS = ["kBET", "BRAS", "iLISI", "CiLISI"]
# Rautenstrauch--Ohler BRAS: cosine distance to all cells in all other batches.
BRAS_IMPLEMENTATION_NOTE = "bras_definition=cosine_mean_other"
OUTPUT_GROUPS = {
    "leiden": ("bio_conservation_leiden", "batch_mixing_leiden"),
    "kmeans": ("bio_conservation", "batch_mixing"),
}


def parse_resolutions(value: str | None) -> list[float] | None:
    if value is None or not value.strip():
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, help="Directory containing embedding .h5ad files.")
    parser.add_argument("--method", required=True, help="Method name to write into output tables.")
    parser.add_argument("--label-key", default="cell_type", help="obs column used as ground-truth labels.")
    parser.add_argument("--batch-key", default="batch_id", help="obs column used as batch labels.")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics and predictions.")
    parser.add_argument("--pattern", default="*.h5ad", help="Glob pattern under --embedding-dir.")
    parser.add_argument("--force", action="store_true", help="Recompute metrics even when per-sample output exists.")
    parser.add_argument(
        "--metric-set",
        default="bio",
        choices=["all", "bio", "batch"],
        help=(
            "Metric group to compute. Leiden only changes bio clustering labels; "
            "batch metrics are available but unchanged except for output location."
        ),
    )
    parser.add_argument(
        "--batch-max-cells",
        type=int,
        default=0,
        help=(
            "Deterministically stratified-subsample batch metrics to at most this many cells. "
            "Use 0 for exact full-data batch metrics."
        ),
    )
    parser.add_argument("--batch-subsample-seed", type=int, default=17, help="Random seed for batch metric subsampling.")
    parser.add_argument("--kbet-neighbors", type=int, default=50, help="Neighbors used for kBET.")
    parser.add_argument("--lisi-perplexity", type=int, default=30, help="Perplexity used for cLISI, iLISI, and CiLISI.")
    parser.add_argument("--graph-neighbors", type=int, default=15, help="Neighbors used for graph connectivity.")
    parser.add_argument(
        "--batch-silhouette-chunk-size",
        type=int,
        default=256,
        help="Chunk size for the BRAS pairwise-distance calculation.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for PyNNDescent.")
    parser.add_argument(
        "--clustering",
        default="leiden",
        choices=["leiden", "kmeans"],
        help="Cluster labels for NMI/HOM/COM/FMI/ARI: Leiden (resolution matched to the number of "
             "cell types) or KMeans (k = number of cell types).",
    )
    parser.add_argument(
        "--leiden-resolutions",
        default=None,
        help="Comma-separated Leiden resolutions. Defaults to 0.1,0.2,...,3.0.",
    )
    parser.add_argument("--leiden-neighbors", type=int, default=15, help="Neighbors used for the Leiden graph.")
    parser.add_argument("--leiden-metric", default="euclidean", help="Distance metric used for the Leiden graph.")
    parser.add_argument("--leiden-random-state", type=int, default=0, help="Random seed for Leiden.")
    parser.add_argument("--leiden-max-resolution", type=float, default=6.0)
    parser.add_argument("--leiden-flavor", default="igraph", help="Leiden backend passed to scanpy.tl.leiden.")
    parser.add_argument(
        "--leiden-no-match-n-labels",
        action="store_true",
        help="Select the resolution with best NMI instead of matching cluster count to label count.",
    )
    return parser.parse_args()


def requested_metrics(args: argparse.Namespace) -> tuple[bool, bool]:
    return args.metric_set in {"all", "bio"}, args.metric_set in {"all", "batch"}


def require_bras_backend() -> None:
    version = getattr(scib_metrics, "__version__", "unknown")
    try:
        release = tuple(int(part) for part in version.split("+")[0].split(".")[:3])
    except (AttributeError, ValueError):
        release = None
    if not callable(getattr(scib_metrics, "bras", None)) or (release is not None and release < (0, 5, 6)):
        raise RuntimeError(
            "This workflow requires scib-metrics>=0.5.6 with scib_metrics.bras; "
            f"found scib-metrics {version}."
        )


def compute_bras(
    x: np.ndarray,
    labels: np.ndarray,
    batches: np.ndarray,
    chunk_size: int,
) -> float:
    finite_rows = np.isfinite(x).all(axis=1)
    if not finite_rows.all():
        raise ValueError(
            "BRAS cosine distance is undefined for "
            f"nonfinite_rows={int((~finite_rows).sum())}"
        )
    zero_norm_rows = np.linalg.norm(x, axis=1) <= 1e-12
    if zero_norm_rows.any():
        raise ValueError(
            "BRAS cosine distance is undefined for "
            f"zero_norm_rows={int(zero_norm_rows.sum())}"
        )
    return float(
        scib_metrics.bras(
            x,
            labels=labels,
            batch=batches,
            metric="cosine",
            chunk_size=chunk_size,
            between_cluster_distances="mean_other",
        )
    )


def as_dense_float_array(x: object) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def valid_neighbor_count(n_cells: int, requested: int) -> int:
    if n_cells < 2:
        return 0
    return min(max(1, requested), n_cells - 1)


def build_knn(x: np.ndarray, n_neighbors: int, n_jobs: int = -1) -> NeighborsResults:
    n_neighbors = valid_neighbor_count(x.shape[0], n_neighbors)
    if n_neighbors < 1:
        raise ValueError("At least two cells are required to build a neighbor graph.")

    n_trees = min(64, 5 + int(round(x.shape[0] ** 0.5 / 20.0)))
    n_iters = max(5, int(round(np.log2(x.shape[0]))))
    index = NNDescent(
        x,
        n_neighbors=n_neighbors,
        random_state=0,
        low_memory=True,
        n_jobs=n_jobs,
        compressed=False,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        metric="euclidean",
    )
    indices, distances = index.neighbor_graph
    return NeighborsResults(indices=indices, distances=distances)


def slice_knn(nn: NeighborsResults, n_neighbors: int) -> NeighborsResults:
    n_neighbors = min(max(1, n_neighbors), nn.indices.shape[1])
    return NeighborsResults(
        indices=nn.indices[:, :n_neighbors],
        distances=nn.distances[:, :n_neighbors],
    )


def leiden_scan_resolution(
    x: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    """Standalone copy of the BioMetrics Leiden scan used for TSv2 comparison."""
    y_true = np.asarray(labels)
    target_k = int(len(np.unique(y_true)))
    resolutions = parse_resolutions(args.leiden_resolutions)
    if not resolutions:
        resolutions = list(np.round(np.arange(0.1, 3.0 + 1e-9, 0.1), 2))

    adata = ad.AnnData(np.asarray(x, dtype=np.float32))
    sc.pp.neighbors(
        adata,
        n_neighbors=args.leiden_neighbors,
        use_rep="X",
        metric=args.leiden_metric,
    )

    records: list[dict[str, float | int]] = []
    all_labels: list[np.ndarray] = []

    def cluster_at(resolution: float) -> np.ndarray:
        key = f"leiden_r_{float(resolution):.4f}"
        try:
            sc.tl.leiden(
                adata,
                resolution=float(resolution),
                random_state=args.leiden_random_state,
                flavor=args.leiden_flavor,
                n_iterations=2,
                directed=False,
                key_added=key,
            )
        except TypeError:
            sc.tl.leiden(
                adata,
                resolution=float(resolution),
                random_state=args.leiden_random_state,
                key_added=key,
            )
        return adata.obs[key].astype(int).to_numpy()

    def evaluate(resolution: float) -> None:
        pred = cluster_at(resolution)
        n_clusters = int(pred.max() + 1) if pred.size else 0
        nmi = float(sklearn.metrics.normalized_mutual_info_score(y_true, pred))
        ari = float(sklearn.metrics.adjusted_rand_score(y_true, pred))
        records.append(
            {
                "resolution": float(resolution),
                "n_clusters": n_clusters,
                "nmi": nmi,
                "ari": ari,
            }
        )
        all_labels.append(pred)

    for resolution in resolutions:
        evaluate(float(resolution))

    if not args.leiden_no_match_n_labels:
        resolution = max(resolutions)
        while max(record["n_clusters"] for record in records) < target_k and resolution < args.leiden_max_resolution:
            resolution = round(float(resolution) + 0.5, 2)
            evaluate(resolution)

    if args.leiden_no_match_n_labels:
        best_i = max(range(len(records)), key=lambda i: records[i]["nmi"])
        selection = "best_nmi"
    else:
        best_i = min(
            range(len(records)),
            key=lambda i: (abs(int(records[i]["n_clusters"]) - target_k), -float(records[i]["nmi"])),
        )
        selection = "match_n_labels"

    best = records[best_i]
    note = (
        f"clustering=leiden;selection={selection};"
        f"resolution={best['resolution']};n_clusters={best['n_clusters']};"
        f"target_n_labels={target_k};scan_n={len(records)}"
    )
    return all_labels[best_i].astype(str), note


def knn_cv_accuracy(x: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    counts = pd.Series(labels).value_counts()
    keep = counts[counts >= 5].index
    mask = np.isin(labels, keep)
    dropped = int((~mask).sum())
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        return float("nan"), f"insufficient_cells_after_dropping_rare_classes;dropped_cells={dropped}"

    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = float(cross_val_score(clf, x[mask], labels[mask], cv=cv).mean())
    note = f"dropped_cells={dropped}" if dropped else ""
    return score, note


def combine_notes(*notes: str) -> str:
    return ";".join(note for note in notes if note)


def make_metric_record(
    sample_id: str,
    method: str,
    metric: str,
    value: float,
    metric_group: str,
    n_cells: int,
    n_dims: int,
    n_labels: int,
    n_batches: int | float,
    note: str = "",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "method": method,
        "metric": metric,
        "value": value,
        "note": note,
        "metric_group": metric_group,
        "n_cells": n_cells,
        "n_dims": n_dims,
        "n_labels": n_labels,
        "n_batches": n_batches,
    }


def safe_add(
    records: list[dict[str, object]],
    sample_id: str,
    method: str,
    metric: str,
    metric_group: str,
    n_cells: int,
    n_dims: int,
    n_labels: int,
    n_batches: int | float,
    func,
    note: str = "",
) -> None:
    try:
        value = float(func())
        metric_note = note
    except Exception as exc:  # noqa: BLE001 - metric libraries raise heterogeneous exceptions.
        value = float("nan")
        metric_note = combine_notes(note, f"metric_failed={type(exc).__name__}:{str(exc).replace(';', ',')}")
    records.append(
        make_metric_record(
            sample_id=sample_id,
            method=method,
            metric=metric,
            value=value,
            metric_group=metric_group,
            n_cells=n_cells,
            n_dims=n_dims,
            n_labels=n_labels,
            n_batches=n_batches,
            note=metric_note,
        )
    )


def metrics_file_complete(path: Path, required_metrics: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        columns = ["metric", "note"] if "BRAS" in required_metrics else ["metric"]
        table = pd.read_csv(path, usecols=columns)
        metrics = set(table["metric"].astype(str))
    except Exception:  # noqa: BLE001 - stale or partial CSVs should be recomputed.
        return False
    if not set(required_metrics).issubset(metrics):
        return False
    if "BRAS" in required_metrics:
        bras_notes = table.loc[table["metric"].astype(str) == "BRAS", "note"].fillna("").astype(str)
        if bras_notes.empty:
            return False
        if not bras_notes.str.contains(BRAS_IMPLEMENTATION_NOTE, regex=False).all():
            return False
        if bras_notes.str.contains("metric_failed=", regex=False).any():
            return False
    return True


def stratified_subsample_indices(
    labels: np.ndarray,
    batches: np.ndarray,
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    n_cells = labels.shape[0]
    if max_cells <= 0 or n_cells <= max_cells:
        return np.arange(n_cells), ""

    label_series = pd.Series(labels, dtype="object").astype(str)
    batch_series = pd.Series(batches, dtype="object").astype(str)
    strata = label_series.str.cat(batch_series, sep="\x1f")
    group_indices = strata.groupby(strata, sort=False).indices
    group_names = np.asarray(list(group_indices.keys()), dtype=object)
    counts = np.asarray([len(group_indices[name]) for name in group_names], dtype=int)

    raw_quota = counts * (max_cells / n_cells)
    quota = np.floor(raw_quota).astype(int)
    quota = np.minimum(quota, counts)

    if len(group_names) <= max_cells:
        quota[(quota == 0) & (counts > 0)] = 1
    else:
        quota[:] = 0
        quota[np.argsort(-counts)[:max_cells]] = 1

    while quota.sum() > max_cells:
        reducible = np.where(quota > 0)[0]
        drop_idx = reducible[np.argmin(raw_quota[reducible])]
        quota[drop_idx] -= 1

    remainder = max_cells - int(quota.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw_quota - np.floor(raw_quota)))
        for idx in fractional_order:
            if remainder == 0:
                break
            room = int(counts[idx] - quota[idx])
            if room <= 0:
                continue
            add_n = min(room, remainder)
            quota[idx] += add_n
            remainder -= add_n

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for name, take_n in zip(group_names, quota):
        if take_n <= 0:
            continue
        idx = np.asarray(group_indices[name], dtype=int)
        selected.append(rng.choice(idx, size=int(take_n), replace=False))

    if not selected:
        return np.arange(n_cells), "subsample_failed_empty_selection"

    out = np.sort(np.concatenate(selected))
    note = f"stratified_subsample=cell_type_x_batch;full_n_cells={n_cells};used_n_cells={out.shape[0]}"
    return out, note


def compute_bio_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    sample_id: str,
    method: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_clusters = len(np.unique(labels))
    records: list[dict[str, object]] = []

    if args.clustering == "kmeans":
        pred_labels = sklearn.cluster.KMeans(
            n_clusters=n_clusters,
            random_state=1,
            n_init=10,
        ).fit(x).labels_.astype(str)
        clustering_note = ""
    else:
        pred_labels, clustering_note = leiden_scan_resolution(x, labels, args)

    def add(metric: str, value: float, note: str = "") -> None:
        records.append(
            make_metric_record(
                sample_id=sample_id,
                method=method,
                metric=metric,
                value=value,
                metric_group="bio_conservation",
                n_cells=x.shape[0],
                n_dims=x.shape[1],
                n_labels=n_clusters,
                n_batches=float("nan"),
                note=note,
            )
        )

    add("NMI", sklearn.metrics.normalized_mutual_info_score(labels, pred_labels), clustering_note)
    add("HOM", sklearn.metrics.homogeneity_score(labels, pred_labels), clustering_note)
    add("COM", sklearn.metrics.completeness_score(labels, pred_labels), clustering_note)
    add("FMI", sklearn.metrics.fowlkes_mallows_score(labels, pred_labels), clustering_note)
    add("ARI", sklearn.metrics.adjusted_rand_score(labels, pred_labels), clustering_note)

    if n_clusters < 2 or x.shape[0] <= n_clusters:
        add("ASW", float("nan"), "insufficient_labels_or_cells")
    else:
        add("ASW", sklearn.metrics.silhouette_score(x, labels, metric="euclidean"))

    if x.shape[0] < 2:
        add("cLISI", float("nan"), "insufficient_cells")
        nn_graph = None
    else:
        max_graph_neighbors = max(int(max(3 * args.lisi_perplexity, args.lisi_perplexity + 1)), args.graph_neighbors)
        nn_graph = build_knn(x, max_graph_neighbors, n_jobs=args.n_jobs)
        safe_add(
            records,
            sample_id,
            method,
            "cLISI",
            "bio_conservation",
            x.shape[0],
            x.shape[1],
            n_clusters,
            float("nan"),
            lambda: scib_metrics.clisi_knn(
                slice_knn(nn_graph, int(max(3 * args.lisi_perplexity, args.lisi_perplexity + 1))),
                labels,
                perplexity=args.lisi_perplexity,
                scale=True,
            ),
        )

    acc, note = knn_cv_accuracy(x, labels)
    add("Acc@kNN", acc, note)

    if nn_graph is None:
        add("GC", float("nan"), "insufficient_cells")
    else:
        safe_add(
            records,
            sample_id,
            method,
            "GC",
            "bio_conservation",
            x.shape[0],
            x.shape[1],
            n_clusters,
            float("nan"),
            lambda: scib_metrics.graph_connectivity(slice_knn(nn_graph, args.graph_neighbors), labels),
        )

    pred_df = pd.DataFrame(
        {
            "barcode": np.arange(x.shape[0]).astype(str),
            "true_label": labels,
            f"{args.clustering}_label": pred_labels,
        }
    )
    pred_df.insert(0, "sample_id", sample_id)
    pred_df.insert(1, "method", method)

    return pd.DataFrame(records), pred_df


def compute_cilisi(
    x: np.ndarray,
    labels: np.ndarray,
    batches: np.ndarray,
    perplexity: int,
    n_jobs: int,
    min_cells: int = 10,
) -> float:
    try:
        lisi_knn = scib_metrics.lisi_knn
    except AttributeError:
        from scib_metrics.metrics import lisi_knn

    all_scores: list[np.ndarray] = []
    for label in np.unique(labels):
        mask = labels == label
        n_cells = int(mask.sum())
        if n_cells < min_cells:
            continue

        batches_sub = batches[mask]
        n_batches = len(np.unique(batches_sub))
        if n_batches < 2:
            continue

        perplexity_sub = min(perplexity, n_cells - 1)
        n_neighbors = int(max(3 * perplexity_sub, perplexity_sub + 1))
        nn = build_knn(x[mask], n_neighbors, n_jobs=n_jobs)
        actual_neighbors = nn.indices.shape[1]
        perplexity_sub = min(perplexity_sub, max(1, actual_neighbors))

        lisi = lisi_knn(nn, batches_sub, perplexity=perplexity_sub)
        scores = (np.asarray(lisi, dtype=float) - 1) / (n_batches - 1)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            continue
        all_scores.append(np.clip(scores, 0, 1))

    if not all_scores:
        return float("nan")
    return float(np.nanmean(np.concatenate(all_scores)))


def compute_batch_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    batches: np.ndarray,
    sample_id: str,
    method: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    n_labels = len(np.unique(labels))
    n_batches = len(np.unique(batches))

    def add_nan(metric: str, note: str) -> None:
        if metric == "BRAS":
            note = combine_notes(note, BRAS_IMPLEMENTATION_NOTE)
        records.append(
            make_metric_record(
                sample_id=sample_id,
                method=method,
                metric=metric,
                value=float("nan"),
                metric_group="batch_mixing",
                n_cells=x.shape[0],
                n_dims=x.shape[1],
                n_labels=n_labels,
                n_batches=n_batches,
                note=note,
            )
        )

    if x.shape[0] < 2:
        for metric in BATCH_METRICS:
            add_nan(metric, "insufficient_cells")
        return pd.DataFrame(records)
    if n_batches < 2:
        for metric in BATCH_METRICS:
            add_nan(metric, "insufficient_batches")
        return pd.DataFrame(records)
    if n_labels < 1:
        for metric in BATCH_METRICS:
            add_nan(metric, "insufficient_labels")
        return pd.DataFrame(records)

    sample_idx, sample_note = stratified_subsample_indices(
        labels=labels,
        batches=batches,
        max_cells=args.batch_max_cells,
        seed=args.batch_subsample_seed,
    )
    xb = x[sample_idx]
    labels_b = labels[sample_idx]
    batches_b = batches[sample_idx]
    n_labels_b = len(np.unique(labels_b))
    n_batches_b = len(np.unique(batches_b))

    if xb.shape[0] < 2 or n_batches_b < 2:
        note = combine_notes(sample_note, "insufficient_batches_after_subsample")
        for metric in BATCH_METRICS:
            add_nan(metric, note)
        return pd.DataFrame(records)

    lisi_neighbors = int(max(3 * args.lisi_perplexity, args.lisi_perplexity + 1))
    nn_batch = build_knn(xb, max(args.kbet_neighbors, lisi_neighbors), n_jobs=args.n_jobs)

    safe_add(
        records,
        sample_id,
        method,
        "kBET",
        "batch_mixing",
        xb.shape[0],
        xb.shape[1],
        n_labels_b,
        n_batches_b,
        lambda: scib_metrics.kbet_per_label(
            slice_knn(nn_batch, args.kbet_neighbors),
            batches_b,
            labels_b,
            alpha=0.05,
            diffusion_n_comps=100,
        ),
        note=sample_note,
    )
    safe_add(
        records,
        sample_id,
        method,
        "BRAS",
        "batch_mixing",
        xb.shape[0],
        xb.shape[1],
        n_labels_b,
        n_batches_b,
        lambda: compute_bras(xb, labels_b, batches_b, args.batch_silhouette_chunk_size),
        note=combine_notes(sample_note, BRAS_IMPLEMENTATION_NOTE),
    )
    safe_add(
        records,
        sample_id,
        method,
        "iLISI",
        "batch_mixing",
        xb.shape[0],
        xb.shape[1],
        n_labels_b,
        n_batches_b,
        lambda: scib_metrics.ilisi_knn(
            slice_knn(nn_batch, lisi_neighbors),
            batches_b,
            perplexity=args.lisi_perplexity,
            scale=True,
        ),
        note=sample_note,
    )
    safe_add(
        records,
        sample_id,
        method,
        "CiLISI",
        "batch_mixing",
        xb.shape[0],
        xb.shape[1],
        n_labels_b,
        n_batches_b,
        lambda: compute_cilisi(
            xb,
            labels_b,
            batches_b,
            perplexity=args.lisi_perplexity,
            n_jobs=args.n_jobs,
        ),
        note=sample_note,
    )
    return pd.DataFrame(records)


def compute_one(
    path: Path,
    method: str,
    label_key: str,
    batch_key: str,
    args: argparse.Namespace,
    compute_bio: bool,
    compute_batch: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    adata = sc.read_h5ad(path)
    if label_key not in adata.obs:
        raise KeyError(f"{path} does not contain obs[{label_key!r}]")
    if compute_batch and batch_key not in adata.obs:
        raise KeyError(f"{path} does not contain obs[{batch_key!r}]")

    obs = adata.obs.copy()
    labels_raw = obs[label_key]
    valid = labels_raw.notna().to_numpy()
    if compute_batch:
        valid &= obs[batch_key].notna().to_numpy()
    if not valid.all():
        adata = adata[valid].copy()
        obs = obs.loc[valid].copy()

    x = as_dense_float_array(adata.X)
    labels = np.asarray(obs[label_key].astype(str))
    batches = np.asarray(obs[batch_key].astype(str)) if compute_batch else np.asarray([], dtype=str)
    sample_id = path.stem

    bio_metrics = None
    pred_df = None
    batch_metrics = None

    if compute_bio:
        bio_metrics, pred_df = compute_bio_metrics(x, labels, sample_id, method, args)
        if pred_df is not None:
            pred_df["barcode"] = adata.obs_names.astype(str)
            pred_df = pred_df[["sample_id", "method", "barcode", "true_label", f"{args.clustering}_label"]]

    if compute_batch:
        batch_metrics = compute_batch_metrics(x, labels, batches, sample_id, method, args)

    return bio_metrics, pred_df, batch_metrics


def write_combined(output_dir: Path, method: str, metric_group: str, frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / f"{method}_{metric_group}_metrics_long.csv"
    combined.to_csv(combined_path, index=False)

    wide = combined.pivot_table(index=["sample_id", "method"], columns="metric", values="value").reset_index()
    wide.to_csv(output_dir / f"{method}_{metric_group}_metrics_wide.csv", index=False)
    print(f"[done] wrote {combined_path}")


def validate_bras_records(frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    bras = combined.loc[combined["metric"].astype(str) == "BRAS"].copy()
    failed = bras["note"].fillna("").astype(str).str.contains("metric_failed=", regex=False)
    values = pd.to_numeric(bras["value"], errors="coerce")
    n_batches = pd.to_numeric(bras["n_batches"], errors="coerce")
    invalid_value = (n_batches >= 2) & (~np.isfinite(values) | (values < 0) | (values > 1))
    invalid = failed | invalid_value
    if invalid.any():
        details = bras.loc[invalid, ["sample_id", "value", "n_batches", "note"]].to_dict("records")
        raise RuntimeError(f"BRAS failed validation for one or more datasets: {details}")


def main() -> None:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    do_bio, do_batch = requested_metrics(args)
    if do_batch:
        require_bras_backend()

    bio_group, batch_group = OUTPUT_GROUPS[args.clustering]
    bio_metrics_dir = output_dir / f"{bio_group}_metrics" / args.method
    pred_dir = output_dir / f"{bio_group}_predictions" / args.method
    batch_metrics_dir = output_dir / f"{batch_group}_metrics" / args.method
    if do_bio:
        bio_metrics_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
    if do_batch:
        batch_metrics_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(embedding_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No embedding files matched {embedding_dir / args.pattern}")

    all_bio_metrics: list[pd.DataFrame] = []
    all_batch_metrics: list[pd.DataFrame] = []
    for path in paths:
        bio_metrics_path = bio_metrics_dir / f"{path.stem}_{args.method}.csv"
        pred_path = pred_dir / f"{path.stem}_{args.method}_{args.clustering}_labels.tsv"
        batch_metrics_path = batch_metrics_dir / f"{path.stem}_{args.method}.csv"

        need_bio = do_bio and (args.force or not metrics_file_complete(bio_metrics_path, BIO_METRICS))
        need_batch = do_batch and (args.force or not metrics_file_complete(batch_metrics_path, BATCH_METRICS))

        if do_bio and not need_bio:
            all_bio_metrics.append(pd.read_csv(bio_metrics_path))
        if do_batch and not need_batch:
            all_batch_metrics.append(pd.read_csv(batch_metrics_path))

        if not need_bio and not need_batch:
            print(f"[skip] {path.stem}")
            continue

        missing_groups = []
        if need_bio:
            missing_groups.append("bio")
        if need_batch:
            missing_groups.append("batch")
        print(f"[run] {path.stem}: {','.join(missing_groups)}")

        bio_metrics, pred, batch_metrics = compute_one(
            path=path,
            method=args.method,
            label_key=args.label_key,
            batch_key=args.batch_key,
            args=args,
            compute_bio=need_bio,
            compute_batch=need_batch,
        )

        if bio_metrics is not None:
            bio_metrics.to_csv(bio_metrics_path, index=False)
            if pred is not None:
                pred.to_csv(pred_path, sep="\t", index=False)
            all_bio_metrics.append(bio_metrics)
        if batch_metrics is not None:
            batch_metrics.to_csv(batch_metrics_path, index=False)
            all_batch_metrics.append(batch_metrics)

    if do_bio:
        write_combined(output_dir, args.method, bio_group, all_bio_metrics)
    if do_batch:
        validate_bras_records(all_batch_metrics)
        write_combined(output_dir, args.method, batch_group, all_batch_metrics)


if __name__ == "__main__":
    main()
