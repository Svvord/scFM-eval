#!/usr/bin/env python3
"""Representation-geometry probes for cell embeddings.

For every embedding file (output of `embed`) paired with the raw-count input it was
computed from, one row of geometry statistics is produced:

  effective dimension      participation ratio PR = (tr C)^2 / tr(C^2) of the
                           coordinate-standardised embedding, and PR / d
  anisotropy               spectral anisotropy sigma_1^2(Z) / ||Z||_F^2 of the
                           uncentred embedding; mean cosine similarity of sampled
                           cell pairs (mixed, within cell type, between cell types,
                           and the within-minus-between contrast)
  expression-neighbourhood preservation
                           chance-corrected R_NX between k-nearest-neighbour graphs
                           built within each batch from analytic Pearson residuals of
                           the raw counts (no dimensionality reduction, no labels) and
                           from the embedding, averaged over k = 15, 30, 50
  intrinsic dimension      TwoNN estimate in the raw and standardised embedding spaces
  partial eta-squared      cell-type and batch effects of the additive per-coordinate
                           model z ~ 1 + cell_type + batch, averaged over coordinates

Cells: all cells when a dataset has at most --max-cells cells, otherwise a seeded
subsample shared by every embedding of that dataset. Pair samples and randomised
algorithms use seeds derived from --seed and the dataset id.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import linalg
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

MIN_LABELED_CELLS = 10
RESULT_COLUMNS = [
    "dataset_id", "method", "embedding_file", "n_cells_total", "n_cells_used", "embedding_dim", "zero_var_dims",
    "pr", "npr",
    "aniso_spec", "aniso_cos", "aniso_cos_within_ct", "aniso_cos_between_ct", "aniso_cos_ct_gap",
    "aniso_pair_n", "aniso_pair_n_within_ct", "aniso_pair_n_between_ct",
    "rnx_k15", "rnx_k30", "rnx_k50", "rnx_mean", "rnx_eligible_cells", "rnx_eligible_cell_fraction", "rnx_n_batches",
    "id_twonn_raw", "id_twonn_raw_n_cells", "id_twonn_z", "id_twonn_z_n_cells",
    "partial_eta2_celltype", "partial_eta2_batch", "partial_eta2_n_dims", "partial_eta2_status",
    "probe_status", "failure_reason", "notes",
]


# --------------------------------------------------------------------------- helpers
def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def seed_for(base: int, *parts: Any) -> int:
    return (int(base) + stable_int("|".join(str(p) for p in parts))) % (2**31 - 1)


def append_note(existing: Any, new: str) -> str:
    existing = "" if existing is None or (isinstance(existing, float) and np.isnan(existing)) else str(existing)
    if not new:
        return existing
    return f"{existing};{new}" if existing else new


# --------------------------------------------------------------------------- data
def load_data(path: Path, max_cells: int, label_key: str, batch_key: str, seed: int) -> dict[str, Any]:
    backed = ad.read_h5ad(path, backed="r")
    n_obs, n_vars = backed.shape
    obs_names = np.asarray(backed.obs_names.astype(str))
    if n_obs > max_cells:
        rng = np.random.default_rng(seed_for(seed, path.stem, "subsample", max_cells))
        selected = np.sort(rng.choice(n_obs, size=max_cells, replace=False).astype(np.int64))
    else:
        selected = np.arange(n_obs, dtype=np.int64)
    adata = backed[selected, :].to_memory()
    backed.file.close()
    adata.obs_names = adata.obs_names.astype(str)
    X = adata.X
    X = sp.csr_matrix(np.asarray(X)) if not sp.issparse(X) else X.tocsr()
    obs = adata.obs.copy()
    obs.index = obs_names[selected]
    labels = obs[label_key].astype("object") if label_key in obs.columns else None
    if batch_key in obs.columns:
        batches = obs[batch_key].astype(str).to_numpy()
        batch_note = f"batch_key={batch_key}"
    else:
        batches = np.array(["batch0"] * len(obs), dtype=object)
        batch_note = f"batch_key={batch_key}_missing;single_batch"
    return {
        "sample_id": path.stem, "path": path, "n_obs": int(n_obs), "n_vars": int(n_vars),
        "obs_names": obs_names, "selected_indices": selected, "selected_obs_names": obs_names[selected],
        "obs": obs, "labels": labels, "batches": batches, "batch_note": batch_note,
        "var_names": np.asarray(adata.var_names.astype(str)), "var": adata.var, "X": X,
    }


def alignment_indices(emb_path: Path, prep: dict[str, Any]) -> tuple[str, Optional[np.ndarray], str]:
    emb = ad.read_h5ad(emb_path, backed="r")
    emb_names = np.asarray(emb.obs_names.astype(str))
    emb.file.close()
    if len(emb_names) == len(prep["obs_names"]) and np.array_equal(emb_names, prep["obs_names"]):
        return "exact_order", prep["selected_indices"].astype(np.int64, copy=False), ""
    if len(np.unique(emb_names)) != len(emb_names):
        return "failed", None, "embedding obs_names are not unique"
    emb_map = {name: i for i, name in enumerate(emb_names)}
    missing = [name for name in prep["selected_obs_names"] if name not in emb_map]
    if missing:
        return "failed", None, f"missing {len(missing)} selected obs_names in embedding"
    return "reordered_by_obs_names", np.asarray([emb_map[name] for name in prep["selected_obs_names"]], dtype=np.int64), ""


def read_dense_rows(path: Path, indices: np.ndarray) -> np.ndarray:
    adata = ad.read_h5ad(path, backed="r")
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        arr = np.empty((0, adata.shape[1]), dtype=np.float32)
    elif np.all(idx[:-1] <= idx[1:]):
        arr = adata.X[idx, :]
    else:
        order = np.argsort(idx)
        inv = np.argsort(order)
        arr = adata.X[idx[order], :][inv, :]
    if sp.issparse(arr):
        arr = arr.toarray()
    arr = np.asarray(arr, dtype=np.float32)
    adata.file.close()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_embedding(Z: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, int]:
    Z = np.asarray(Z, dtype=np.float32)
    mean = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = Z - mean
    std = centered.std(axis=0, dtype=np.float64).astype(np.float32)
    zero = ~(std > eps)
    std_safe = std.copy()
    std_safe[zero] = 1.0
    Zs = centered / std_safe
    if np.any(zero):
        Zs[:, zero] = 0.0
    Zs = np.nan_to_num(Zs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return Zs, int(np.sum(zero))


# --------------------------------------------------------------------------- effective dimension
def participation_ratio(Zs: np.ndarray, exact_max_dim: int, hutchinson_probes: int, seed: int) -> tuple[float, float, str]:
    n, d = Zs.shape
    if n < 2 or d < 1:
        return np.nan, np.nan, "insufficient_data"
    Zc = Zs - Zs.mean(axis=0, dtype=np.float64).astype(np.float32)
    trace = float(np.sum(Zc * Zc, dtype=np.float64) / n)
    if trace <= 0:
        return 0.0, 0.0, "zero_trace"
    if d <= exact_max_dim:
        cov = (Zc.T @ Zc).astype(np.float64) / float(n)
        fro2 = float(np.sum(cov * cov))
        method = "exact_covariance_frobenius"
        del cov
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(d, hutchinson_probes), replace=True)
        Zv = Zc @ signs
        Cv = (Zc.T @ Zv) / float(n)
        fro2 = float(np.mean(np.sum(Cv * Cv, axis=0, dtype=np.float64)))
        method = f"hutchinson_frobenius_q{hutchinson_probes}"
        del signs, Zv, Cv
    if fro2 <= 0:
        return np.nan, np.nan, method + ";nonpositive_frobenius"
    pr = (trace * trace) / fro2
    return float(pr), float(pr / float(d)), method


# --------------------------------------------------------------------------- anisotropy
def sample_mixed_pairs(n_cells: int, pair_n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
    if n_cells < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), "insufficient_cells"
    ii = rng.integers(0, n_cells, size=pair_n, dtype=np.int64)
    jj = rng.integers(0, n_cells, size=pair_n, dtype=np.int64)
    same = ii == jj
    while np.any(same):
        jj[same] = rng.integers(0, n_cells, size=int(np.sum(same)), dtype=np.int64)
        same = ii == jj
    return ii, jj, "ok"


def celltype_groups(labels: pd.Series) -> dict[str, np.ndarray]:
    clean = labels.astype("object")
    valid = clean.notna() & (clean.astype(str).str.len() > 0) & (clean.astype(str).str.lower() != "nan")
    groups: dict[str, np.ndarray] = {}
    for label, idx in clean[valid].astype(str).groupby(clean[valid].astype(str)).groups.items():
        groups[str(label)] = np.asarray(idx, dtype=np.int64)
    return groups


def sample_within_celltype_pairs(groups, pair_n, rng):
    eligible = {label: idx for label, idx in groups.items() if len(idx) >= 2}
    if not eligible:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), "no_celltype_with_at_least_two_cells"
    labels = np.asarray(sorted(eligible), dtype=object)
    chosen = rng.choice(labels, size=pair_n, replace=True)
    ii = np.empty(pair_n, dtype=np.int64)
    jj = np.empty(pair_n, dtype=np.int64)
    for label in labels:
        loc = np.where(chosen == label)[0]
        if len(loc) == 0:
            continue
        idx = eligible[str(label)]
        ii[loc] = rng.choice(idx, size=len(loc), replace=True)
        jj[loc] = rng.choice(idx, size=len(loc), replace=True)
        same = ii[loc] == jj[loc]
        while np.any(same):
            replace_loc = loc[same]
            jj[replace_loc] = rng.choice(idx, size=len(replace_loc), replace=True)
            same = ii[loc] == jj[loc]
    return ii, jj, "celltype_balanced"


def sample_between_celltype_pairs(groups, pair_n, rng):
    eligible = {label: idx for label, idx in groups.items() if len(idx) >= 1}
    if len(eligible) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), "fewer_than_two_celltypes"
    labels = np.asarray(sorted(eligible), dtype=object)
    left_labels = rng.choice(labels, size=pair_n, replace=True)
    right_labels = rng.choice(labels, size=pair_n, replace=True)
    same = left_labels == right_labels
    while np.any(same):
        right_labels[same] = rng.choice(labels, size=int(np.sum(same)), replace=True)
        same = left_labels == right_labels
    ii = np.empty(pair_n, dtype=np.int64)
    jj = np.empty(pair_n, dtype=np.int64)
    for label in labels:
        left_loc = np.where(left_labels == label)[0]
        if len(left_loc):
            ii[left_loc] = rng.choice(eligible[str(label)], size=len(left_loc), replace=True)
        right_loc = np.where(right_labels == label)[0]
        if len(right_loc):
            jj[right_loc] = rng.choice(eligible[str(label)], size=len(right_loc), replace=True)
    return ii, jj, "celltype_balanced"


def build_anisotropy_pair_sets(labels: Optional[pd.Series], n_cells: int, pair_n: int, seed: int):
    rng = np.random.default_rng(seed)
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mixed_i, mixed_j, _ = sample_mixed_pairs(n_cells, pair_n, rng)
    pairs["mixed"] = (mixed_i, mixed_j)
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if labels is None:
        pairs["within_ct"] = empty
        pairs["between_ct"] = empty
        return pairs, "missing_cell_type"
    groups = celltype_groups(labels.reset_index(drop=True))
    within_i, within_j, within_status = sample_within_celltype_pairs(groups, pair_n, rng)
    between_i, between_j, between_status = sample_between_celltype_pairs(groups, pair_n, rng)
    pairs["within_ct"] = (within_i, within_j)
    pairs["between_ct"] = (between_i, between_j)
    return pairs, f"within={within_status};between={between_status};n_celltypes={len(groups)}"


def anisotropy_cosine_for_pairs(Z: np.ndarray, pairs) -> tuple[float, int, str]:
    ii, jj = pairs
    pair_n = int(len(ii))
    if pair_n == 0:
        return np.nan, 0, "no_pairs"
    norms = np.linalg.norm(Z, axis=1).astype(np.float64)
    total = 0.0
    count = 0
    for start in range(0, pair_n, 20000):
        stop = min(pair_n, start + 20000)
        dots = np.sum(Z[ii[start:stop], :] * Z[jj[start:stop], :], axis=1, dtype=np.float64)
        den = norms[ii[start:stop]] * norms[jj[start:stop]]
        ok = den > 0
        total += float(np.sum(dots[ok] / den[ok]))
        count += int(np.sum(ok))
    aniso_cos = total / count if count else np.nan
    note = "fixed_pairs" if count == pair_n else f"fixed_pairs;valid_nonzero_pairs={count}/{pair_n}"
    return float(aniso_cos), int(count), note


def anisotropy_spectral(Z: np.ndarray, seed: int) -> tuple[float, str]:
    n = Z.shape[0]
    if n < 2:
        return np.nan, "insufficient_cells"
    total_lambda = float(np.sum(Z * Z, dtype=np.float64) / n)
    if total_lambda <= 0:
        return np.nan, "nonpositive_total_energy"
    try:
        _, s, _ = randomized_svd(Z, n_components=1, n_iter=5, random_state=seed)
        lambda1 = float(s[0] * s[0] / n)
        return float(lambda1 / total_lambda), "randomized_svd_top1"
    except Exception as exc:  # noqa: BLE001
        return np.nan, f"aniso_spec_failed:{type(exc).__name__}:{exc}"


# --------------------------------------------------------------------------- neighbours / TwoNN
def compute_neighbors(X: np.ndarray, k: int, seed: int, n_jobs: int, exact_max_cells: int):
    n = X.shape[0]
    if n <= 1:
        return np.empty((n, 0), dtype=np.int64), np.empty((n, 0), dtype=np.float32), "insufficient_cells"
    k_eff = min(k, n - 1)
    if n <= exact_max_cells:
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", algorithm="auto", n_jobs=n_jobs)
        nn.fit(X)
        dists, inds = nn.kneighbors(X, return_distance=True)
        source = "sklearn_exact"
    else:
        import pynndescent

        index = pynndescent.NNDescent(X, n_neighbors=k_eff + 1, metric="euclidean", random_state=seed,
                                      n_jobs=max(1, min(int(n_jobs), 4)) if n_jobs > 0 else 4, low_memory=True, verbose=False)
        inds, dists = index.neighbor_graph
        source = "pynndescent_approx"
    clean_inds = np.full((n, k_eff), -1, dtype=np.int64)
    clean_dists = np.full((n, k_eff), np.nan, dtype=np.float32)
    for i in range(n):
        row_i = np.asarray(inds[i], dtype=np.int64)
        row_d = np.asarray(dists[i], dtype=np.float32)
        mask = row_i != i
        row_i = row_i[mask][:k_eff]
        row_d = row_d[mask][:k_eff]
        clean_inds[i, : len(row_i)] = row_i
        clean_dists[i, : len(row_d)] = row_d
    return clean_inds, clean_dists, source


def twonn_id(neighbor_dists: np.ndarray) -> tuple[float, int]:
    if neighbor_dists.shape[1] < 2:
        return np.nan, 0
    r1 = neighbor_dists[:, 0].astype(np.float64)
    r2 = neighbor_dists[:, 1].astype(np.float64)
    ok = np.isfinite(r1) & np.isfinite(r2) & (r1 > 0) & (r2 > r1)
    if int(np.sum(ok)) < 10:
        return np.nan, int(np.sum(ok))
    mu = r2[ok] / r1[ok]
    denom = float(np.sum(np.log(mu)))
    if denom <= 0:
        return np.nan, int(np.sum(ok))
    return float(len(mu) / denom), int(len(mu))


# --------------------------------------------------------------------------- expression-neighbourhood preservation
def validate_raw_counts(X: sp.csr_matrix) -> None:
    if X.data.size:
        min_value = float(np.min(X.data))
        probe = X.data if X.data.size <= 1_000_000 else X.data[:: max(1, X.data.size // 1_000_000)]
        max_integer_error = float(np.max(np.abs(probe - np.rint(probe))))
    else:
        min_value = max_integer_error = 0.0
    if min_value < 0 or max_integer_error > 1e-6:
        raise ValueError("expression-neighbourhood preservation requires raw non-negative integer counts in X; "
                         f"observed min={min_value:.6g}, max integer error={max_integer_error:.6g}")


def technical_gene_mask(var: pd.DataFrame, var_names: np.ndarray, X: sp.csr_matrix) -> np.ndarray:
    upper = np.char.upper(var_names.astype(str))
    detected = np.asarray(X.getnnz(axis=0)).ravel().astype(np.int64)
    totals = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    mt = np.char.startswith(upper, "MT-")
    ercc = np.char.startswith(upper, "ERCC-")
    if "mt" in var.columns:
        mt = mt | var["mt"].fillna(False).astype(bool).to_numpy()
    if "ercc" in var.columns:
        ercc = ercc | var["ercc"].fillna(False).astype(bool).to_numpy()
    ribosomal = np.char.startswith(upper, "RPL") | np.char.startswith(upper, "RPS")
    excluded = mt | ercc | ribosomal | (detected < 10) | (totals < 10)
    return ~excluded


def pearson_residual_block(X_batch: sp.csr_matrix, full_library: np.ndarray, gene_indices: np.ndarray, theta: float, clip: float) -> np.ndarray:
    observed = X_batch[:, gene_indices].toarray().astype(np.float32, copy=False)
    grand_total = float(np.sum(full_library, dtype=np.float64))
    if grand_total <= 0:
        raise ValueError("batch has zero total counts")
    gene_total = np.asarray(observed.sum(axis=0, dtype=np.float64)).ravel()
    probability = gene_total / grand_total
    mu = full_library.astype(np.float64, copy=False)[:, None] * probability[None, :]
    denom = np.sqrt(mu + (mu * mu) / float(theta))
    residual = np.divide(observed.astype(np.float64, copy=False) - mu, denom, out=np.zeros_like(mu, dtype=np.float64), where=denom > 0)
    np.clip(residual, -clip, clip, out=residual)
    return residual.astype(np.float32, copy=False)


def select_residual_genes(X: sp.csr_matrix, batches: np.ndarray, candidate_mask: np.ndarray, eligible_batches: list[str],
                          n_genes: int, theta: float, chunk_size: int) -> np.ndarray:
    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
    if len(candidate_indices) < n_genes:
        raise ValueError(f"only {len(candidate_indices)} eligible genes remain; requested {n_genes}")
    percentile_rows = []
    variance_rows = []
    for batch in eligible_batches:
        idx = np.flatnonzero(batches == batch).astype(np.int64)
        Xb = X[idx, :].tocsr()
        library = np.asarray(Xb.sum(axis=1)).ravel().astype(np.float64)
        variances = np.zeros(len(candidate_indices), dtype=np.float64)
        for start in range(0, len(candidate_indices), chunk_size):
            stop = min(start + chunk_size, len(candidate_indices))
            residual = pearson_residual_block(Xb, library, candidate_indices[start:stop], theta=theta, clip=math.sqrt(len(idx)))
            variances[start:stop] = np.var(residual, axis=0, ddof=1, dtype=np.float64)
            del residual
        percentile_rows.append(rankdata(variances, method="average") / float(len(variances)))
        variance_rows.append(variances)
        del Xb
    percentiles = np.vstack(percentile_rows)
    variances = np.vstack(variance_rows)
    detection = np.asarray(X[:, candidate_indices].getnnz(axis=0)).ravel().astype(np.int64)
    ranking = pd.DataFrame({
        "gene_index": candidate_indices,
        "median_pct": np.median(percentiles, axis=0),
        "mean_pct": np.mean(percentiles, axis=0),
        "detected": detection,
        "gene_order": np.arange(len(candidate_indices)),
    }).sort_values(["median_pct", "mean_pct", "detected", "gene_order"], ascending=[False, False, False, True], kind="mergesort")
    return ranking.head(n_genes)["gene_index"].to_numpy(dtype=np.int64)


def clean_knn_rows(indices: np.ndarray, n_cells: int, k: int) -> np.ndarray:
    clean = np.full((n_cells, k), -1, dtype=np.int64)
    for i in range(n_cells):
        seen: set[int] = set()
        values: list[int] = []
        for raw in indices[i]:
            val = int(raw)
            if val == i or val < 0 or val >= n_cells or val in seen:
                continue
            seen.add(val)
            values.append(val)
            if len(values) == k:
                break
        if len(values) != k:
            raise RuntimeError(f"kNN row {i} contained only {len(values)} valid unique neighbours; expected {k}")
        clean[i, :] = values
    return clean


def compute_knn(X: np.ndarray, k: int, seed: int, n_jobs: int, exact_max_cells: int) -> tuple[np.ndarray, str]:
    X = np.asarray(X, dtype=np.float32)
    n = int(X.shape[0])
    if n <= k:
        raise ValueError(f"need more than k={k} cells, observed n={n}")
    if n <= exact_max_cells:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute", n_jobs=n_jobs)
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)
        backend = "sklearn_brute_exact"
    else:
        import pynndescent

        build_k = min(n - 1, max(2 * k, k + 16))
        index = pynndescent.NNDescent(X, n_neighbors=build_k, metric="euclidean", random_state=seed,
                                      n_jobs=max(1, min(int(n_jobs), 4)) if n_jobs > 0 else 4, low_memory=True,
                                      n_trees=16, n_iters=15, verbose=False)
        indices, _ = index.neighbor_graph
        backend = f"pynndescent_approx_buildk{build_k}"
    return clean_knn_rows(np.asarray(indices), n, k), backend


def graph_overlap_rnx(ref: np.ndarray, emb: np.ndarray, ks: Iterable[int]) -> dict[int, float]:
    n = int(ref.shape[0])
    result: dict[int, float] = {}
    for k in ks:
        overlap_sum = 0.0
        for start in range(0, n, 1024):
            stop = min(start + 1024, n)
            a = ref[start:stop, :k]
            b = emb[start:stop, :k]
            overlap = np.any(a[:, :, None] == b[:, None, :], axis=2).sum(axis=1)
            overlap_sum += float(np.sum(overlap, dtype=np.float64))
        qnx = overlap_sum / float(n * k)
        result[int(k)] = float(((n - 1.0) * qnx - k) / (n - 1.0 - k))
    return result


def build_expression_reference(prep: dict[str, Any], args: argparse.Namespace):
    """Per-batch reference kNN graphs from analytic Pearson residuals. Returns (graphs, batch_qc, note)."""
    X = prep["X"]
    validate_raw_counts(X)
    batches = prep["batches"]
    batch_sizes = pd.Series(batches).value_counts().sort_index()
    eligible = [str(b) for b, n in batch_sizes.items() if int(n) >= args.min_batch_cells]
    if not eligible:
        raise ValueError(f"no batch has at least {args.min_batch_cells} cells; cannot build the within-batch reference")
    candidate_mask = technical_gene_mask(prep["var"], prep["var_names"], X)
    genes = select_residual_genes(X, batches, candidate_mask, eligible, args.n_genes, args.theta, args.gene_chunk_size)
    graphs: dict[str, np.ndarray] = {}
    qc = []
    for batch in eligible:
        idx = np.flatnonzero(batches == batch).astype(np.int64)
        Xb = X[idx, :].tocsr()
        library = np.asarray(Xb.sum(axis=1)).ravel().astype(np.float64)
        residual = pearson_residual_block(Xb, library, genes, theta=args.theta, clip=math.sqrt(len(idx)))
        graphs[batch], backend = compute_knn(residual, max(args.rnx_ks), seed=seed_for(args.seed, prep["sample_id"], batch, "native_reference"),
                                             n_jobs=args.n_jobs, exact_max_cells=args.exact_knn_max_cells)
        qc.append({"batch": batch, "n_cells": int(len(idx)), "reference_knn_backend": backend})
        del Xb, residual
    n_eligible_cells = int(sum(q["n_cells"] for q in qc))
    note = f"{prep['batch_note']};n_genes={len(genes)};eligible_batches={len(eligible)}/{len(batch_sizes)};eligible_cells={n_eligible_cells}"
    return graphs, qc, n_eligible_cells, note


def expression_rnx(Z: np.ndarray, prep: dict[str, Any], graphs: dict[str, np.ndarray], method: str, args: argparse.Namespace):
    rows = []
    for batch, ref in graphs.items():
        idx = np.flatnonzero(prep["batches"] == batch).astype(np.int64)
        emb_graph, backend = compute_knn(Z[idx, :], max(args.rnx_ks), seed=seed_for(args.seed, prep["sample_id"], batch, method),
                                         n_jobs=args.n_jobs, exact_max_cells=args.exact_knn_max_cells)
        for k, rnx in graph_overlap_rnx(ref, emb_graph, args.rnx_ks).items():
            rows.append({"dataset_id": prep["sample_id"], "method": method, "batch": batch, "n_cells": int(len(idx)), "k": int(k),
                         "RNX": rnx, "embedding_knn_backend": backend})
        del emb_graph
    df = pd.DataFrame(rows)
    per_k = {}
    for k, group in df.groupby("k", sort=True):
        per_k[int(k)] = float(np.average(group["RNX"], weights=group["n_cells"].to_numpy(dtype=float)))
    return per_k, df


# --------------------------------------------------------------------------- partial eta-squared
def orthonormal_basis(design: np.ndarray) -> tuple[np.ndarray, int]:
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        return np.empty((design.shape[0], 0), dtype=np.float64), 0
    q, r, _ = linalg.qr(np.asarray(design, dtype=np.float64), mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0 or not np.isfinite(diagonal[0]) or diagonal[0] == 0:
        return np.empty((design.shape[0], 0), dtype=np.float64), 0
    tolerance = float(max(design.shape) * np.finfo(np.float64).eps * diagonal[0])
    rank = int(np.sum(diagonal > tolerance))
    return q[:, :rank], rank


def dummy_matrix(series: pd.Series) -> np.ndarray:
    return pd.get_dummies(series.astype(str), drop_first=True, dtype=float).to_numpy(dtype=np.float64)


def design_matrix(n: int, *parts: np.ndarray) -> np.ndarray:
    matrices = [np.ones((n, 1), dtype=np.float64)]
    matrices.extend(part for part in parts if part.shape[1] > 0)
    return np.column_stack(matrices)


def projection_ss(q: np.ndarray, values: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    projected = q.T @ values
    return np.einsum("ij,ij->j", projected, projected, optimize=True)


def partial_eta2(Zs: np.ndarray, labels: Optional[pd.Series], batches: np.ndarray, single_batch: bool) -> tuple[float, float, int, str]:
    """Mean over coordinates of SS_effect / (SS_effect + SSE_full) for cell type and batch."""
    if labels is None:
        return np.nan, np.nan, 0, "missing_cell_type"
    cell_raw = labels.astype("object")
    valid = cell_raw.notna().to_numpy(dtype=bool)
    n = int(valid.sum())
    if n < MIN_LABELED_CELLS:
        return np.nan, np.nan, 0, "insufficient_labeled_cells"
    cell = pd.Series(cell_raw.to_numpy()[valid]).astype(str)
    batch = pd.Series(np.asarray(batches)[valid]).astype(str)
    y = np.asarray(Zs[valid, :], dtype=np.float64)
    cell_dm = dummy_matrix(cell)
    batch_dm = dummy_matrix(batch)
    q_full, rank_full = orthonormal_basis(design_matrix(n, cell_dm, batch_dm))
    q_no_cell, rank_no_cell = orthonormal_basis(design_matrix(n, batch_dm))
    q_no_batch, rank_no_batch = orthonormal_basis(design_matrix(n, cell_dm))
    residual_df = n - rank_full
    if residual_df <= 0:
        return np.nan, np.nan, int(y.shape[1]), "no_residual_df"
    total = np.einsum("ij,ij->j", y, y, optimize=True)
    sse_full = total - projection_ss(q_full, y)
    centered = y - y.mean(axis=0, keepdims=True)
    sst = np.einsum("ij,ij->j", centered, centered, optimize=True)
    tolerance = 1e-8 * np.maximum.reduce([np.ones_like(total), np.abs(total), np.abs(sst)])

    def effect(q_reduced: np.ndarray, estimable: int) -> float:
        if estimable <= 0:
            return np.nan
        sse_reduced = total - projection_ss(q_reduced, y)
        ss_effect = sse_reduced - sse_full
        ok = (sst > 1e-12) & np.isfinite(ss_effect) & np.isfinite(sse_full) & (sse_full >= -tolerance) & (ss_effect >= -tolerance)
        ss_e = np.where(ss_effect < 0, 0.0, ss_effect)
        sse_f = np.where(sse_full < 0, 0.0, sse_full)
        denominator = ss_e + sse_f
        ok &= denominator > 1e-12
        if not np.any(ok):
            return np.nan
        return float(np.mean(ss_e[ok] / denominator[ok]))

    eta_cell = effect(q_no_cell, rank_full - rank_no_cell) if cell.nunique() >= 2 else np.nan
    eta_batch = np.nan if single_batch or batch.nunique() < 2 else effect(q_no_batch, rank_full - rank_no_batch)
    status_parts = [f"n_labeled={n}", f"n_celltypes={cell.nunique()}", f"n_batches={batch.nunique()}"]
    if rank_full - rank_no_cell < max(0, cell.nunique() - 1) or (not single_batch and rank_full - rank_no_batch < max(0, batch.nunique() - 1)):
        status_parts.append("partially_aliased_effects")
    if single_batch:
        status_parts.append("batch_effect_not_available")
    return eta_cell, eta_batch, int(y.shape[1]), ";".join(status_parts)


# --------------------------------------------------------------------------- driver
def pair_files(embeddings: list[Path], data_files: list[Path]) -> list[tuple[Path, Path]]:
    if len(embeddings) == 1 and len(data_files) == 1:
        return [(embeddings[0], data_files[0])]
    by_stem = {p.stem: p for p in data_files}
    pairs = []
    for emb in embeddings:
        if emb.stem not in by_stem:
            raise FileNotFoundError(f"no input data file named {emb.stem}.h5ad for embedding {emb.name}")
        pairs.append((emb, by_stem[emb.stem]))
    return pairs


def process_pair(emb_path: Path, data_path: Path, method: str, args: argparse.Namespace):
    prep = load_data(data_path, args.max_cells, args.label_key, args.batch_key, args.seed)
    single_batch = "single_batch" in prep["batch_note"]
    row: dict[str, Any] = {c: np.nan for c in RESULT_COLUMNS}
    row.update({"dataset_id": prep["sample_id"], "method": method, "embedding_file": str(emb_path.name),
                "n_cells_total": prep["n_obs"], "n_cells_used": int(len(prep["selected_indices"])),
                "probe_status": "completed", "failure_reason": "", "notes": ""})
    rnx_rows = pd.DataFrame()
    try:
        status, emb_indices, align_note = alignment_indices(emb_path, prep)
        if emb_indices is None:
            raise RuntimeError(f"obs_alignment_failed:{align_note}")
        Zraw = read_dense_rows(emb_path, emb_indices)
        row["embedding_dim"] = int(Zraw.shape[1])
        Zs, zero_var_dims = standardize_embedding(Zraw)
        row["zero_var_dims"] = int(zero_var_dims)
        notes = [f"obs_alignment={status}"]

        pr, npr, pr_note = participation_ratio(Zs, args.exact_pr_max_dim, args.hutchinson_probes, seed_for(args.seed, prep["sample_id"], method, "pr"))
        row["pr"], row["npr"] = pr, npr
        notes.append(f"pr={pr_note}")

        pairs, pair_note = build_anisotropy_pair_sets(prep["labels"], Zraw.shape[0], args.pair_n, seed_for(args.seed, prep["sample_id"], "anisotropy_pairs"))
        row["aniso_cos"], row["aniso_pair_n"], _ = anisotropy_cosine_for_pairs(Zraw, pairs["mixed"])
        row["aniso_cos_within_ct"], row["aniso_pair_n_within_ct"], _ = anisotropy_cosine_for_pairs(Zraw, pairs["within_ct"])
        row["aniso_cos_between_ct"], row["aniso_pair_n_between_ct"], _ = anisotropy_cosine_for_pairs(Zraw, pairs["between_ct"])
        w, b = row["aniso_cos_within_ct"], row["aniso_cos_between_ct"]
        row["aniso_cos_ct_gap"] = w - b if pd.notna(w) and pd.notna(b) else np.nan
        row["aniso_spec"], spec_note = anisotropy_spectral(Zraw, seed_for(args.seed, prep["sample_id"], method, "anisotropy_spec"))
        notes.append(f"aniso_pairs={pair_note};aniso_spec={spec_note}")

        _, raw_dists, knn_source = compute_neighbors(Zraw, args.knn_k, seed_for(args.seed, prep["sample_id"], "p3_knn"), args.n_jobs, args.exact_knn_max_cells)
        row["id_twonn_raw"], row["id_twonn_raw_n_cells"] = twonn_id(raw_dists)
        _, z_dists, _ = compute_neighbors(Zs, args.knn_k, seed_for(args.seed, prep["sample_id"], method, "twonn_knn"), args.n_jobs, args.exact_knn_max_cells)
        row["id_twonn_z"], row["id_twonn_z_n_cells"] = twonn_id(z_dists)
        notes.append(f"twonn_knn={knn_source}")

        try:
            graphs, qc, n_eligible, rnx_note = build_expression_reference(prep, args)
            per_k, rnx_rows = expression_rnx(Zraw, prep, graphs, method, args)
            for k, value in per_k.items():
                row[f"rnx_k{k}"] = value
            row["rnx_mean"] = float(np.mean(list(per_k.values())))
            row["rnx_eligible_cells"] = n_eligible
            row["rnx_eligible_cell_fraction"] = n_eligible / float(len(prep["batches"]))
            row["rnx_n_batches"] = len(graphs)
            notes.append(f"rnx:{rnx_note}")
            del graphs
        except Exception as exc:  # noqa: BLE001
            notes.append(f"rnx_failed={type(exc).__name__}:{str(exc).replace(';', ',')}")

        eta_cell, eta_batch, n_dims, eta_status = partial_eta2(Zs, prep["labels"], prep["batches"], single_batch)
        row["partial_eta2_celltype"], row["partial_eta2_batch"] = eta_cell, eta_batch
        row["partial_eta2_n_dims"], row["partial_eta2_status"] = n_dims, eta_status
        row["notes"] = ";".join(notes)
        del Zraw, Zs
    except Exception as exc:  # noqa: BLE001
        logging.exception("geometry probes failed for %s / %s", method, prep["sample_id"])
        row["probe_status"] = "failed"
        row["failure_reason"] = f"{type(exc).__name__}:{exc}"
    gc.collect()
    return row, rnx_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding-dir", required=True, help="directory with embedding .h5ad files")
    ap.add_argument("--data-dir", required=True, help="directory with the raw-count input .h5ad files (matched by file name)")
    ap.add_argument("--pattern", default="*.h5ad")
    ap.add_argument("--method", required=True, help="method label written into the tables")
    ap.add_argument("--label-key", default="cell_type")
    ap.add_argument("--batch-key", default="batch_id")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-cells", type=int, default=20000)
    ap.add_argument("--pair-n", type=int, default=100000)
    ap.add_argument("--knn-k", type=int, default=30, help="neighbours computed for the TwoNN estimate")
    ap.add_argument("--rnx-ks", default="15,30,50")
    ap.add_argument("--n-genes", type=int, default=2000)
    ap.add_argument("--theta", type=float, default=100.0)
    ap.add_argument("--min-batch-cells", type=int, default=101)
    ap.add_argument("--gene-chunk-size", type=int, default=2000)
    ap.add_argument("--exact-knn-max-cells", type=int, default=5000)
    ap.add_argument("--exact-pr-max-dim", type=int, default=1600)
    ap.add_argument("--hutchinson-probes", type=int, default=128)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.rnx_ks = [int(x) for x in str(args.rnx_ks).split(",") if x.strip()]
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[geometry] %(message)s")
    args = parse_args()
    embeddings = sorted(Path(args.embedding_dir).glob(args.pattern))
    data_files = sorted(Path(args.data_dir).glob("*.h5ad"))
    if not embeddings:
        raise FileNotFoundError(f"no embedding files matched {args.embedding_dir}/{args.pattern}")
    if not data_files:
        raise FileNotFoundError(f"no input data files in {args.data_dir}")
    out = Path(args.output_dir)
    per_sample = out / "geometry" / args.method
    per_sample.mkdir(parents=True, exist_ok=True)
    rows = []
    for emb_path, data_path in pair_files(embeddings, data_files):
        logging.info("%s | %s", args.method, emb_path.stem)
        row, rnx_rows = process_pair(emb_path, data_path, args.method, args)
        rows.append(row)
        pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS).to_csv(per_sample / f"{emb_path.stem}_{args.method}.csv", index=False)
        if len(rnx_rows):
            rnx_rows.to_csv(per_sample / f"{emb_path.stem}_{args.method}_rnx_batches.csv", index=False)
        logging.info("%s | %s: %s", args.method, emb_path.stem, row["probe_status"] if row["probe_status"] == "completed" else row["failure_reason"])
    table = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    combined = out / f"{args.method}_geometry.csv"
    table.to_csv(combined, index=False)
    logging.info("wrote %s", combined)


if __name__ == "__main__":
    main()
