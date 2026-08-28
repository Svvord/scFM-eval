#!/usr/bin/env python
"""Frozen-embedding label transfer: fit a lightweight classifier on reference cell
embeddings, then predict labels for query cell embeddings.

    classifier.py fit     --embedding ref.h5ad --label-key cell_type --classifier logreg \
                          --method scgpt --out <model_dir>
    classifier.py predict --embedding query.h5ad --model <model_dir> --out-prefix <id>

Classifiers
  prototype  class-mean prototypes; query cells are scored by softmax(-cosine distance)
             (the few-shot protocol of the manuscript)
  knn        k-nearest neighbours in the reference (cosine metric, distance-weighted
             votes; k is capped at the reference size)
  logreg     z-scored embeddings + multinomial logistic regression (linear probe)
  mlp        the manuscript's post-hoc MLP head (celltype_annotation_finetune.py)

The model directory holds meta.json plus model.npz (prototype/knn/logreg) or an mlp/
directory. Prediction writes <prefix>_predicted_probs.tsv (barcode x class) and
<prefix>_predicted_labels.tsv (column `predicted_label`).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.special import softmax

CLASSIFIERS = ("prototype", "knn", "logreg", "mlp")
HERE = os.path.dirname(os.path.abspath(__file__))
POSTHOC_FIT = os.path.join(os.path.dirname(HERE), "celltype_annotation_finetune.py")
POSTHOC_PREDICT = os.path.join(os.path.dirname(HERE), "celltype_annotation_predict.py")


def load_embedding(path, label_key=None):
    adata = sc.read_h5ad(path)
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    barcodes = adata.obs["barcode"].astype(str).tolist() if "barcode" in adata.obs else adata.obs_names.astype(str).tolist()
    labels = None
    if label_key is not None:
        if label_key not in adata.obs:
            sys.exit("label column '{}' not found in obs of {}".format(label_key, path))
        labels = adata.obs[label_key].astype(str).to_numpy()
    return X, barcodes, labels


# ----------------------------------------------------------------------------- fit
def fit(args):
    if args.classifier not in CLASSIFIERS:
        sys.exit("unknown classifier '{}'; choose from {}".format(args.classifier, ", ".join(CLASSIFIERS)))
    X, _, y = load_embedding(args.embedding, args.label_key)
    classes = sorted(set(y))
    counts = pd.Series(y).value_counts()
    meta = {
        "format": "scfoundry_transfer/1",
        "classifier": args.classifier,
        "method": args.method,
        "label_key": args.label_key,
        "classes": classes,
        "dim": int(X.shape[1]),
        "n_reference": int(X.shape[0]),
        "cells_per_class": {c: int(counts[c]) for c in classes},
        "hyperparameters": {},
    }
    os.makedirs(args.out, exist_ok=True)

    if args.classifier == "prototype":
        prototypes = np.stack([X[y == c].mean(axis=0) for c in classes])
        np.savez_compressed(os.path.join(args.out, "model.npz"), prototypes=prototypes, classes=np.array(classes))

    elif args.classifier == "knn":
        k = int(min(args.knn_k, X.shape[0]))
        meta["hyperparameters"] = {"k": k, "metric": "cosine", "weights": "distance"}
        y_idx = np.array([classes.index(c) for c in y], dtype=np.int64)   # integer codes: no object arrays in the npz
        np.savez_compressed(os.path.join(args.out, "model.npz"), X=X, y=y_idx, classes=np.array(classes))

    elif args.classifier == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        clf = LogisticRegression(max_iter=args.max_iter, C=args.C, random_state=args.seed)
        clf.fit(Z, y)
        assert list(clf.classes_) == classes
        meta["hyperparameters"] = {"C": args.C, "max_iter": args.max_iter, "scaling": "z-score",
                                   "n_iter": int(np.max(clf.n_iter_))}
        np.savez_compressed(os.path.join(args.out, "model.npz"), mean=scaler.mean_.astype(np.float32),
                            scale=scaler.scale_.astype(np.float32), coef=clf.coef_.astype(np.float32),
                            intercept=clf.intercept_.astype(np.float32), classes=np.array(classes))

    elif args.classifier == "mlp":
        # The post-hoc head keeps a stratified validation split for early stopping
        # (floor(n_class * 0.2) cells per class); refuse references too small to yield one.
        n_eval = int(sum(min(int(n * 0.2), n - 1) for n in counts))
        if n_eval == 0:
            sys.exit("mlp needs a validation split, i.e. at least one class with >= 5 reference cells "
                     "(largest class has {}); use --classifier prototype, knn or logreg for such small "
                     "references".format(int(counts.max())))
        cmd = [sys.executable, POSTHOC_FIT, "--data_path", args.embedding, "--method_name", "mlp",
               "--label_key", args.label_key, "--seed", str(args.seed)]
        print("[classifier] running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        produced = "mlp_finetuned_model"
        if not os.path.isdir(produced):
            sys.exit("post-hoc classifier did not produce {}".format(produced))
        dest = os.path.join(args.out, "mlp")
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.move(produced, dest)
        meta["hyperparameters"] = {"script": os.path.basename(POSTHOC_FIT)}

    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[classifier] fitted {} on {} cells x {} dims, {} classes -> {}".format(
        args.classifier, X.shape[0], X.shape[1], len(classes), args.out))


# ------------------------------------------------------------------------- predict
def predict(args):
    with open(os.path.join(args.model, "meta.json")) as fh:
        meta = json.load(fh)
    clf = meta["classifier"]
    classes = list(meta["classes"])
    X, barcodes, _ = load_embedding(args.embedding)
    if X.shape[1] != meta["dim"]:
        sys.exit("embedding dimension {} does not match the fitted model ({}, method {})".format(
            X.shape[1], meta["dim"], meta.get("method")))

    if clf == "mlp":
        cmd = [sys.executable, POSTHOC_PREDICT, "--method_name", "mlp", "--data_path", args.embedding,
               "--model_dir", os.path.join(args.model, "mlp")]
        print("[classifier] running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        probs = pd.read_csv("mlp_predictions.tsv", sep="\t", index_col=0)
        probs.index = barcodes
        probs = probs[[c for c in probs.columns]]
    else:
        m = np.load(os.path.join(args.model, "model.npz"), allow_pickle=False)
        if clf == "prototype":
            dist = cdist(X, m["prototypes"], metric="cosine")
            P = softmax(-dist, axis=-1)
        elif clf == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            k = int(meta["hyperparameters"]["k"])
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance").fit(m["X"], m["y"])
            P = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
            P[:, knn.classes_] = knn.predict_proba(X)          # classes_ are the integer codes present in the reference
        elif clf == "logreg":
            Z = (X - m["mean"]) / m["scale"]
            z = Z @ m["coef"].T + m["intercept"]
            if len(classes) == 2:               # scikit-learn stores one column for binary problems
                p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
                P = np.stack([1.0 - p1, p1], axis=1)
            else:
                P = softmax(z, axis=-1)
        else:
            sys.exit("unknown classifier in meta.json: {}".format(clf))
        probs = pd.DataFrame(P, index=barcodes, columns=classes)

    probs.to_csv("{}_predicted_probs.tsv".format(args.out_prefix), sep="\t")
    labels = pd.DataFrame({"predicted_label": probs.idxmax(axis=1)}, index=probs.index)
    labels.to_csv("{}_predicted_labels.tsv".format(args.out_prefix), sep="\t")
    print("[classifier] predicted {} cells with {} ({} classes)".format(len(labels), clf, len(classes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--embedding", required=True)
    f.add_argument("--label-key", default="cell_type")
    f.add_argument("--classifier", default="logreg", choices=CLASSIFIERS)
    f.add_argument("--method", default="")
    f.add_argument("--out", required=True)
    f.add_argument("--knn-k", type=int, default=15)
    f.add_argument("--C", type=float, default=1.0)
    f.add_argument("--max-iter", type=int, default=2000)
    f.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("predict")
    p.add_argument("--embedding", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    fit(args) if args.cmd == "fit" else predict(args)


if __name__ == "__main__":
    main()
