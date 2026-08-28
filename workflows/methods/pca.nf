params.pca_n_top_genes = 2000
params.pca_n_comps = 50
params.emb_results_dir = "results"

process embed_by_pca {

    tag "${id}"

    label 'cpu_task'

    container 'housy17/scllms:latest'

    publishDir "${params.emb_results_dir}/embeddings/pca", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("PCA"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import scanpy as sc
    import pandas as pd

    adata = sc.read_h5ad("${raw_h5ad}")
    original_obs = adata.obs.copy()
    original_spatial = adata.obsm["spatial"].copy() if "spatial" in adata.obsm else None

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=${params.pca_n_top_genes},
        flavor="seurat",
        subset=False,
    )

    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=${params.pca_n_comps}, svd_solver="arpack", random_state=0)

    embedding = adata.obsm["X_pca"]
    adata_embedding = sc.AnnData(
        X=embedding,
        obs=original_obs,
        var=pd.DataFrame(index=[f"V{i+1}" for i in range(embedding.shape[1])]),
    )
    if original_spatial is not None:
        adata_embedding.obsm["spatial"] = original_spatial
    adata_embedding.write("pca_embeddings.h5ad")
    """
}
