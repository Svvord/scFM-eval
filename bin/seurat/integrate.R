#!/usr/bin/env Rscript
# Seurat v5 batch integration -> integrated cell embedding.
#
# Reads a Python-exported MatrixMarket bundle (features x cells) plus a metadata
# table, runs Seurat v5 `IntegrateLayers` with ONLY the batch covariate (no cell
# type), and writes the integrated low-dimensional embedding as a CSV.
#
# Usage:
#   Rscript integrate.R <method> <n_hvg> <n_pcs> <in_dir> <out_csv>
#     method : cca | rpca | harmony
#     in_dir : directory holding matrix.mtx, features.tsv, barcodes.tsv, meta.csv
#
# Only `meta$batch` is used to split layers; cell type never enters the model.

suppressMessages({
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
  library(future)
})

# IntegrateLayers parallelizes across batches via `future` and exports large
# closures; run in-process and lift the global-size guard so it doesn't abort.
plan("sequential")
options(future.globals.maxSize = 16 * 1024^3)

args   <- commandArgs(trailingOnly = TRUE)
method <- args[1]
n_hvg  <- as.integer(args[2])
n_pcs  <- as.integer(args[3])
in_dir <- args[4]
out    <- args[5]

set.seed(0)

# ---- load the exported matrix (features x cells) ----
m        <- readMM(file.path(in_dir, "matrix.mtx"))
features <- readLines(file.path(in_dir, "features.tsv"))
cells    <- readLines(file.path(in_dir, "barcodes.tsv"))
features <- make.unique(features)
rownames(m) <- features
colnames(m) <- cells
m <- as(m, "CsparseMatrix")

meta <- read.csv(file.path(in_dir, "meta.csv"), stringsAsFactors = FALSE)
rownames(meta) <- meta$cell
meta <- meta[cells, , drop = FALSE]

obj <- CreateSeuratObject(counts = m, meta.data = meta)

# ---- split into per-batch layers (the ONLY supervision) ----
n_batches <- length(unique(obj$batch))
cat(sprintf("[integrate.R] method=%s cells=%d genes=%d batches=%d\n",
            method, ncol(obj), nrow(obj), n_batches))
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$batch)

# ---- standard per-layer preprocessing + joint PCA ----
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, nfeatures = n_hvg, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = n_pcs, verbose = FALSE)

# ---- integrate ----
new_red <- paste0("integrated_", method)
if (n_batches < 2) {
  # nothing to integrate; fall back to the uncorrected PCA
  cat("[integrate.R] <2 batches; returning uncorrected PCA.\n")
  emb <- Embeddings(obj, reduction = "pca")
} else if (method == "harmony") {
  obj <- IntegrateLayers(object = obj, method = HarmonyIntegration, dims = 1:n_pcs,
                         orig.reduction = "pca", new.reduction = new_red, verbose = FALSE)
  emb <- Embeddings(obj, reduction = new_red)
} else {
  # Anchor-based CCA/RPCA project each batch onto n_pcs dimensions, so every batch must
  # contain more than n_pcs cells; fail early with an actionable message instead of
  # letting Seurat abort deep inside IntegrateLayers.
  min_cells <- min(table(obj$batch))
  if (min_cells <= n_pcs) {
    stop(sprintf(paste0("[integrate.R] %s cannot integrate batches with <= n_pcs=%d cells ",
                        "(smallest batch: %d cells). Merge or drop tiny batches, choose a coarser ",
                        "batch column (--batch_key), or use harmony."),
                 method, n_pcs, min_cells))
  }
  mfun <- switch(method,
    cca  = CCAIntegration,
    rpca = RPCAIntegration,
    stop(paste("unknown method:", method))
  )
  # Anchor-based CCA/RPCA abort when a batch pair yields fewer anchors than
  # k.weight (default 100) — common with small/uneven batches (e.g. an 11-batch
  # blood split, or batches of ~100 cells). Retry with a decreasing k.weight
  # until it succeeds instead of failing the whole tissue.
  integrated <- NULL
  for (kw in c(100, 50, 30, 20, 10, 5)) {
    integrated <- tryCatch(
      IntegrateLayers(object = obj, method = mfun, orig.reduction = "pca", dims = 1:n_pcs,
                      new.reduction = new_red, k.weight = kw, verbose = FALSE),
      error = function(e) {
        cat(sprintf("[integrate.R] %s k.weight=%d failed: %s\n", method, kw, conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(integrated)) {
      cat(sprintf("[integrate.R] %s integrated with k.weight=%d\n", method, kw))
      break
    }
  }
  if (is.null(integrated)) stop(sprintf("IntegrateLayers(%s) failed for all k.weight values", method))
  obj <- integrated
  emb <- Embeddings(obj, reduction = new_red)
}

write.csv(emb, file = out)
cat(sprintf("[integrate.R] wrote %s  (%d x %d)\n", out, nrow(emb), ncol(emb)))
