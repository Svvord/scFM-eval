#!/usr/bin/env python3
"""Advisory check of .h5ad files against the scFoundry input contract.

Reads every file matching --pattern under --data-dir, prints a report per file
(OK / WARN / FAIL lines) and writes it to --output-dir/<stem>_check.txt. Nothing is
modified and the exit status is always 0: the report is the product. FAIL marks a
definite contract violation (a task will error or produce nonsense), WARN a heuristic
worth a look, INFO a fact about the file.

Requires anndata, numpy, scipy (the housy17/scllms image).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import scipy.sparse as sp

INTEGRATION = {"scgpt_integrated", "scvi_denovo", "harmony", "seurat_cca", "seurat_rpca"}
NEEDS_ENSEMBL = {"geneformer", "scprint"}
NEEDS_SPATIAL = {"novae"}
ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}(\.\d+)?$")


class Report:
    def __init__(self, name):
        self.name = name
        self.lines = []
        self.counts = {"OK": 0, "INFO": 0, "WARN": 0, "FAIL": 0}
        self.limits = []   # what the file cannot be used for as it stands

    def add(self, level, msg):
        self.counts[level] += 1
        self.lines.append("  [{:<4}] {}".format(level, msg))

    ok = lambda self, m: self.add("OK", m)
    info = lambda self, m: self.add("INFO", m)
    warn = lambda self, m: self.add("WARN", m)
    fail = lambda self, m: self.add("FAIL", m)

    def render(self):
        c = self.counts
        head = "== {} ==".format(self.name)
        summary = "  {} ok, {} warning{}, {} problem{}".format(
            c["OK"], c["WARN"], "" if c["WARN"] == 1 else "s", c["FAIL"], "" if c["FAIL"] == 1 else "s")
        if c["FAIL"]:
            verdict = ["  fix the problems before running a task on this file"]
        else:
            verdict = ["  ready for embed" + (", but read the warnings" if c["WARN"] else "")]
            verdict += ["    without {}".format(l) for l in self.limits]
        return "\n".join([head] + self.lines + [summary] + verdict)


def cells(n):
    return "{} cell{}".format(n, "" if n == 1 else "s")


def sample_values(X, n=200_000):
    if sp.issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()
    if data.size == 0:
        return data
    if data.size > n:
        step = max(1, data.size // n)
        data = data[::step][:n]
    return np.asarray(data)


def check_file(path, args):
    import anndata as ad

    rep = Report(os.path.basename(path))
    try:
        adata = ad.read_h5ad(path)
    except Exception as exc:  # noqa: BLE001
        rep.fail("cannot be read as AnnData: {}: {}".format(type(exc).__name__, exc))
        return rep
    n_obs, n_vars = adata.shape
    rep.info("{:,} cells x {:,} genes".format(n_obs, n_vars))
    method = (args.method or "").lower()

    # ---------------------------------------------------------------- X: raw counts
    X = adata.X
    if X is None:
        rep.fail("adata.X is empty")
    else:
        vals = sample_values(X)
        kind = "sparse ({})".format(type(X).__name__) if sp.issparse(X) else "dense {}".format(np.asarray(X).dtype)
        if vals.size == 0:
            rep.fail("adata.X holds no values")
        else:
            vmax = float(np.nanmax(vals))
            vmin = float(np.nanmin(vals))
            if not np.isfinite(vals).all():
                rep.fail("adata.X contains NaN or infinite values")
            if vmin < 0:
                rep.fail("adata.X contains negative values ({:.3g}): scaled data, not counts".format(vmin))
            integer = bool(np.allclose(vals, np.round(vals), atol=1e-6))
            alt = ["layers['{}']".format(k) for k in ("counts", "raw_counts", "count") if k in adata.layers]
            if adata.raw is not None:
                alt.append("adata.raw")
            if integer:
                rep.ok("adata.X is integer-valued (raw counts), {}, maximum {:g}".format(kind, vmax))
                if vmax < 30:
                    rep.warn("maximum count is only {:g}: fine for a shallow assay, suspicious otherwise".format(vmax))
            elif vmax < 30:
                rep.fail("adata.X is not integer-valued and its maximum is {:.2f}: this looks log-normalised. "
                         "Models expect raw counts in X{}".format(
                             vmax, "; raw counts appear to be in {}".format(", ".join(alt)) if alt else ""))
            else:
                rep.warn("adata.X is not integer-valued (maximum {:.3g}): acceptable for corrected counts such as "
                         "SoupX output, wrong for normalised data{}".format(
                             vmax, "; note that {} also exist".format(", ".join(alt)) if alt else ""))

    # ---------------------------------------------------------------- genes
    if n_vars < 15_000:
        rep.warn("only {:,} genes: models match genes against their own vocabulary, so an HVG-subset "
                 "input is scored on your gene selection rather than on the model (fine for a targeted panel)".format(n_vars))
    else:
        rep.ok("{:,} genes: full transcriptome".format(n_vars))
    if not adata.var_names.is_unique:
        rep.fail("adata.var_names are not unique ({} duplicates); run adata.var_names_make_unique()".format(
            int(adata.var_names.duplicated().sum())))
    else:
        rep.ok("gene index is unique")
    ens_frac = float(np.mean([bool(ENSEMBL_RE.match(str(g))) for g in adata.var_names[:5000]])) if n_vars else 0.0
    if ens_frac > 0.9:
        rep.warn("the gene index consists of Ensembl IDs ({:.0%}); the contract wants HGNC symbols with Ensembl "
                 "IDs only as a fallback for genes without a symbol".format(ens_frac))
    elif ens_frac > 0:
        rep.info("{:.1%} of gene names are Ensembl IDs (fallback for genes without a symbol)".format(ens_frac))
    if "gene_symbol" in adata.var.columns:
        if adata.var["gene_symbol"].astype(str).duplicated().any():
            rep.warn("var['gene_symbol'] has duplicated values")
        if not (adata.var["gene_symbol"].astype(str).values == adata.var_names.astype(str).values).all():
            rep.warn("var['gene_symbol'] differs from the gene index; the index is what most models read")
        else:
            rep.ok("var['gene_symbol'] present and equal to the gene index")
    else:
        rep.fail("var['gene_symbol'] is missing")
    if "ensembl_id" in adata.var.columns:
        rep.ok("var['ensembl_id'] present")
    elif method in NEEDS_ENSEMBL:
        rep.fail("var['ensembl_id'] is missing and {} needs it".format(method))
    else:
        rep.info("var['ensembl_id'] is missing")
        rep.limits.append("var['ensembl_id'] you cannot run geneformer or scprint")

    # ---------------------------------------------------------------- cells
    if not adata.obs_names.is_unique:
        rep.fail("adata.obs_names are not unique; run adata.obs_names_make_unique()")
    if "barcode" in adata.obs.columns:
        bc = adata.obs["barcode"].astype(str)
        if bc.duplicated().any():
            rep.fail("obs['barcode'] has {} duplicates; prediction tables are indexed by it".format(int(bc.duplicated().sum())))
        elif not (bc.values == adata.obs_names.astype(str).values).all():
            rep.warn("obs['barcode'] differs from obs_names; outputs are indexed by barcode, inputs matched by obs_names")
        else:
            rep.ok("obs['barcode'] present, unique and equal to obs_names")
    else:
        rep.fail("obs['barcode'] is missing; set adata.obs['barcode'] = adata.obs_names")

    # ---------------------------------------------------------------- labels
    key = args.label_key
    if key in adata.obs.columns:
        lab = adata.obs[key]
        n_na = int(lab.isna().sum())
        counts = lab.dropna().astype(str).value_counts()
        rep.ok("obs['{}'] present: {} class{}, smallest has {}, largest {}{}".format(
            key, len(counts), "" if len(counts) == 1 else "es", cells(int(counts.min()) if len(counts) else 0),
            cells(int(counts.max()) if len(counts) else 0), ", {} unlabelled".format(cells(n_na)) if n_na else ""))
        if len(counts) and counts.min() < 5:
            rep.warn("some classes have fewer than 5 cells: as a transfer reference, use --knn-k no larger than "
                     "the smallest class, and the mlp classifier will refuse to hold out a validation split")
        if len(counts) == 1:
            rep.warn("obs['{}'] has a single class".format(key))
    elif args.role == "reference":
        rep.fail("obs['{}'] is missing: a reference for transfer/finetune needs labels (or pass --label-key)".format(key))
    else:
        rep.info("obs['{}'] is missing (not needed to embed; pass --label-key if the labels are in another column)".format(key))
        rep.limits.append("obs['{}'] you cannot run benchmark or use the file as a transfer/finetune reference; "
                          "geometry runs, but its cell-type statistics (within/between-type anisotropy, "
                          "partial eta-squared) stay empty".format(key))

    # ---------------------------------------------------------------- batches
    bkey = args.batch_key
    if bkey in adata.obs.columns:
        counts = adata.obs[bkey].astype(str).value_counts()
        rep.ok("obs['{}'] present: {} batch{}, smallest has {}".format(
            bkey, len(counts), "" if len(counts) == 1 else "es", cells(int(counts.min()))))
        if len(counts) == 1:
            rep.info("a single batch: integration methods have nothing to correct and batch metrics are undefined")
        elif counts.min() <= 30:
            msg = ("the smallest batch has {}: seurat_cca and seurat_rpca cannot integrate batches of 30 cells or "
                   "fewer (harmony can)".format(cells(int(counts.min()))))
            (rep.warn if method in ("seurat_cca", "seurat_rpca") else rep.info)(msg)
    elif method in INTEGRATION:
        rep.fail("obs['{}'] is missing and {} needs batch labels (or pass --batch-key)".format(bkey, method))
    else:
        rep.info("obs['{}'] is missing (not needed to embed; pass --batch-key if the batches are in another column)".format(bkey))
        rep.limits.append("obs['{}'] the integration methods, the batch-mixing metrics and the batch terms of "
                          "geometry treat the data as one batch".format(bkey))

    # ---------------------------------------------------------------- spatial
    if "spatial" in adata.obsm:
        rep.info("obsm['spatial'] present ({} coordinates per cell)".format(adata.obsm["spatial"].shape[1]))
    elif method in NEEDS_SPATIAL:
        rep.fail("obsm['spatial'] is missing and {} is a spatial model".format(method))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pattern", default="*.h5ad")
    ap.add_argument("--method", default="", help="method the file is meant for (adds method-specific checks)")
    ap.add_argument("--label-key", default="cell_type")
    ap.add_argument("--batch-key", default="batch_id")
    ap.add_argument("--role", default="", choices=["", "reference", "query"],
                    help="reference: labels are required; query: labels are optional")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not files:
        sys.exit("no files matching {} under {}".format(args.pattern, args.data_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    for path in files:
        rep = check_file(path, args)
        text = rep.render()
        print(text)
        print()
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(args.output_dir, stem + "_check.txt"), "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
