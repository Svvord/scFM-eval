// ===================== Seurat v5 / Harmony batch integration ===================== //
// Classical batch-integration baselines that return a corrected cell embedding,
// using ONLY the batch label (obs['batch_id']); no cell type enters the model.
// All three run through Seurat v5 `IntegrateLayers` in the satijalab/seurat:5.5.0
// image (which also ships the `harmony` package), so one image covers:
//   - harmony      -> HarmonyIntegration
//   - seurat_cca   -> CCAIntegration
//   - seurat_rpca  -> RPCAIntegration
// The Seurat image has no h5ad reader, so a small Python step converts the input
// AnnData to a MatrixMarket bundle (scPRINT-style preprocess -> run -> postprocess),
// and a final Python step assembles the embedding h5ad with the original obs.

params.batch_key     = "batch_id"
params.seurat_n_hvg  = 2000
params.seurat_n_pcs  = 30
params.emb_results_dir = "results"

process preprocess_for_seurat {

    tag "${id}"

    label "cpu_task"

    container "housy17/scllms:latest"

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path(raw_h5ad), path("seurat_in")

    script:
    """
    #!/usr/bin/env python
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.io import mmwrite
    from scipy.sparse import csr_matrix, issparse

    adata = sc.read_h5ad("${raw_h5ad}")
    if "${params.batch_key}" not in adata.obs:
        # No batch annotation: treat the dataset as one batch (the R step then returns
        # the uncorrected PCA embedding). Use --batch_key to point at the right column.
        print("WARNING: obs['${params.batch_key}'] not found; treating all cells as a single batch.")
        adata.obs["${params.batch_key}"] = "batch0"

    adata.var_names_make_unique()
    X = adata.X
    if not issparse(X):
        X = csr_matrix(X)

    os.makedirs("seurat_in", exist_ok=True)
    # MatrixMarket expects features x cells for Seurat
    mmwrite("seurat_in/matrix.mtx", X.T.tocoo())

    n = adata.n_obs
    cells = [f"cell{i}" for i in range(n)]            # munge-proof positional ids
    with open("seurat_in/barcodes.tsv", "w") as fh:
        fh.write("\\n".join(cells) + "\\n")
    with open("seurat_in/features.tsv", "w") as fh:
        fh.write("\\n".join(map(str, adata.var_names)) + "\\n")

    # ONLY the batch label is exported (sanitized so it is a clean factor level)
    codes = adata.obs["${params.batch_key}"].astype("category").cat.codes.values
    meta = pd.DataFrame({"cell": cells, "batch": ["b" + str(int(c)) for c in codes]})
    meta.to_csv("seurat_in/meta.csv", index=False)
    """
}

process run_seurat_integration {

    tag "${id}:${method}"

    label "cpu_task"

    container "satijalab/seurat:5.5.0"

    input:
    tuple val(id), path(raw_h5ad), path(seurat_in)
    val method

    output:
    tuple val(id), path(raw_h5ad), path("embedding.csv")

    script:
    """
    Rscript /code/seurat/integrate.R ${method} ${params.seurat_n_hvg} ${params.seurat_n_pcs} ${seurat_in} embedding.csv
    """
}

// One finalize process per method: identical body (shared bin script), differing
// only in the static publishDir folder + display label. (publishDir cannot read
// an input val, so the folder must be baked into each process.)
process finalize_harmony {
    tag "${id}"
    label "cpu_task"
    container "housy17/scllms:latest"
    publishDir "${params.emb_results_dir}/embeddings/harmony", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean
    input:
    tuple val(id), path(raw_h5ad), path(embedding_csv)
    output:
    tuple val(id), val("Harmony"), path("*_embeddings.h5ad")
    script:
    """
    python /code/seurat/assemble_embedding.py ${embedding_csv} ${raw_h5ad} harmony_integrated_embeddings.h5ad
    """
}

process finalize_seurat_cca {
    tag "${id}"
    label "cpu_task"
    container "housy17/scllms:latest"
    publishDir "${params.emb_results_dir}/embeddings/seurat_cca", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean
    input:
    tuple val(id), path(raw_h5ad), path(embedding_csv)
    output:
    tuple val(id), val("Seurat CCA"), path("*_embeddings.h5ad")
    script:
    """
    python /code/seurat/assemble_embedding.py ${embedding_csv} ${raw_h5ad} seurat_cca_integrated_embeddings.h5ad
    """
}

process finalize_seurat_rpca {
    tag "${id}"
    label "cpu_task"
    container "housy17/scllms:latest"
    publishDir "${params.emb_results_dir}/embeddings/seurat_rpca", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean
    input:
    tuple val(id), path(raw_h5ad), path(embedding_csv)
    output:
    tuple val(id), val("Seurat RPCA"), path("*_embeddings.h5ad")
    script:
    """
    python /code/seurat/assemble_embedding.py ${embedding_csv} ${raw_h5ad} seurat_rpca_integrated_embeddings.h5ad
    """
}

workflow integrate_by_harmony {
    take:
        Raw_H5ad_Channel
    main:
        preprocess_for_seurat(Raw_H5ad_Channel)
        run_seurat_integration(preprocess_for_seurat.out, "harmony")
        finalize_harmony(run_seurat_integration.out)
    emit:
        finalize_harmony.out
}

workflow integrate_by_seurat_cca {
    take:
        Raw_H5ad_Channel
    main:
        preprocess_for_seurat(Raw_H5ad_Channel)
        run_seurat_integration(preprocess_for_seurat.out, "cca")
        finalize_seurat_cca(run_seurat_integration.out)
    emit:
        finalize_seurat_cca.out
}

workflow integrate_by_seurat_rpca {
    take:
        Raw_H5ad_Channel
    main:
        preprocess_for_seurat(Raw_H5ad_Channel)
        run_seurat_integration(preprocess_for_seurat.out, "rpca")
        finalize_seurat_rpca(run_seurat_integration.out)
    emit:
        finalize_seurat_rpca.out
}
