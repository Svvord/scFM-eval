#!/usr/bin/env python
"""Assemble a Seurat/Harmony integrated embedding into the standard embedding h5ad.

Reads the integrated embedding CSV (rows = positional 'cell<i>' ids, written by
integrate.R) and the original AnnData, and writes an h5ad with the embedding in X
and the original obs carried through (so batch_id + cell_type survive for scoring).

Usage: assemble_embedding.py <embedding_csv> <raw_h5ad> <out_h5ad>
"""
import sys

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

embedding_csv, raw_h5ad, out_h5ad = sys.argv[1], sys.argv[2], sys.argv[3]

emb = pd.read_csv(embedding_csv, index_col=0)
# rows are 'cell<i>' positional ids -> restore original AnnData order
order = emb.index.str.replace("cell", "", regex=False).astype(int).values
emb = emb.iloc[np.argsort(order)]
X = emb.values.astype(np.float32)

adata = sc.read_h5ad(raw_h5ad)
assert X.shape[0] == adata.n_obs, (X.shape, adata.n_obs)

out = ad.AnnData(
    X=X,
    obs=adata.obs.copy(),
    var=pd.DataFrame(index=[f"V{i+1}" for i in range(X.shape[1])]),
)
if "spatial" in adata.obsm:
    out.obsm["spatial"] = adata.obsm["spatial"]
out.write_h5ad(out_h5ad)
