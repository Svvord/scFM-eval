params.model = "SCimilarity/model_v1.1"
params.batch_size = 2048
params.scimilarity_use_gpu = true
params.emb_results_dir = "results"

process preprocess_for_scimilarity {

    tag "${id}"

    container "housy17/scimilarity:latest"

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("temp_scimilarity.h5ad")

    script:
    """
    #!/usr/bin/env python3
    import scanpy as sc
    import numpy as np
    import pandas as pd
    import anndata
    import warnings
    from scipy.sparse import csr_matrix
    from scimilarity import CellAnnotation

    adata = sc.read_h5ad("${raw_h5ad}")

    if "counts" in adata.layers:
        if isinstance(adata.layers["counts"], np.ndarray):
            adata.layers["counts"] = csr_matrix(adata.layers["counts"])
    else:
        if isinstance(adata.X, np.ndarray):
            adata.layers["counts"] = csr_matrix(adata.X)
        else:
            adata.layers["counts"] = adata.X.copy()
    adata.X = csr_matrix(adata.layers["counts"].shape)

    model_path = "/data/model_weights/${params.model}"
    ca = CellAnnotation(model_path=model_path)
    reference_genes = list(ca.gene_order)
    reference_gene_set = set(reference_genes)
    adata_gene_set = set(adata.var_names)
    gene_overlap = len(reference_gene_set & adata_gene_set)

    if gene_overlap < 5000:
        adata_copy = adata.copy()
        n_cells = adata.n_obs
        missing_genes = sorted(reference_gene_set - adata_gene_set)
        warnings.warn(
            f"SCimilarity gene overlap is {gene_overlap} (<5000) for ${id}; "
            f"applying fallback by padding {len(missing_genes)} missing reference genes with zero counts.",
            RuntimeWarning,
        )
        X_missing = csr_matrix((n_cells, len(missing_genes)))
        missing_adata = anndata.AnnData(
            X=X_missing,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=missing_genes),
        )
        missing_adata.layers["counts"] = X_missing.copy()
        adata = anndata.concat([adata_copy, missing_adata], axis=1)
        adata.obs = adata_copy.obs
        adata.obsm = adata_copy.obsm

    adata.write_h5ad("temp_scimilarity.h5ad")
    """
}

process _embed_by_scimilarity {

    tag "${id}"

    label 'gpu_task'

    container "housy17/scimilarity:latest"

    publishDir "${params.emb_results_dir}/embeddings/scimilarity", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("SCimilarity"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python3
    import scanpy as sc
    from scimilarity.utils import lognorm_counts, align_dataset
    from scimilarity import CellEmbedding
    from pathlib import Path
    import os
    model_path = "/data/model_weights/${params.model}"
    use_gpu = "${params.scimilarity_use_gpu}".strip().lower() in {"1", "true", "yes", "y"}
    ce = CellEmbedding(model_path, use_gpu=use_gpu)

    adams = sc.read("${raw_h5ad}")
    adams = align_dataset(adams, ce.gene_order)
    adams = lognorm_counts(adams)

    embeddings = ce.get_embeddings(adams.X, buffer_size=${params.batch_size})

    var_names = [f'V{i+1}' for i in range(embeddings.shape[1])]
    import pandas as pd
    adata_emb = sc.AnnData(X=embeddings, obs=adams.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adams.obsm:
        adata_emb.obsm['spatial'] = adams.obsm['spatial']
    adata_emb.write(f"scimilarity_embeddings.h5ad")
    """
}

workflow embed_by_scimilarity {
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_scimilarity(Raw_H5ad_Channel)
        _embed_by_scimilarity(preprocess_for_scimilarity.out)

    emit:
        _embed_by_scimilarity.out
}
