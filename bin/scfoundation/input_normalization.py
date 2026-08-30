"""Turn raw counts into the input scFoundation was pretrained on.

Per cell: total-count normalisation to 1e4 followed by log1p -- exactly what
get_embedding.py computes for zero-shot embedding when --pre_normalized F is given
(`np.log1p(x / x.sum() * 1e4)`). The fine-tuning and prediction scripts share this
helper so that a fine-tuned model sees the same representation as the frozen one.
"""
import numpy as np
from scipy.sparse import issparse


def normalize_log1p(X, target_sum=1e4):
    X = X.toarray() if issparse(X) else np.asarray(X)
    X = X.astype(np.float32, copy=True)
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0          # an empty cell stays all-zero
    np.multiply(X, target_sum / totals, out=X)
    return np.log1p(X, out=X)
