params.scvi_model = "scVI/Census2024-07-01-HomoSapiens"
params.batch_size = 1024
params.scvi_use_gpu = true
params.emb_results_dir = "results"

process embed_by_scvi {

    tag "${id}"

    label 'gpu_task'

    container 'scverse/scvi-tools:py3.11-cu12-runtime-stable'

    publishDir "${params.emb_results_dir}/embeddings/scvi", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("scVI"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import os

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scvi
    import torch
    from scipy.sparse import csr_matrix, issparse

    model_dir = "/data/model_weights/${params.scvi_model}"
    model_pt = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(
            f"Missing scVI Census model: {model_pt}. "
            "Run: nextflow run download_model_weights.nf --method scvi"
        )

    adata = ad.read_h5ad("${raw_h5ad}")
    original_obs = adata.obs.copy()
    original_spatial = adata.obsm["spatial"].copy() if "spatial" in adata.obsm else None

    if "ensembl_id" not in adata.var:
        raise KeyError("scVI Census projection requires adata.var['ensembl_id']")

    query_var = adata.var.copy()
    query_var.index = query_var["ensembl_id"].astype(str)
    valid_var = query_var.index.notna() & (query_var.index != "") & ~query_var.index.duplicated()
    query = ad.AnnData(
        X=adata.X[:, valid_var].copy(),
        obs=adata.obs.copy(),
        var=query_var.loc[valid_var].copy(),
    )
    query.var_names = query.var.index.astype(str)
    query.obs["batch"] = "unassigned"

    for key in list(query.obs.columns):
        if key.startswith("_scvi"):
            del query.obs[key]
    for key in list(query.uns.keys()):
        if key.startswith("_scvi"):
            del query.uns[key]
    for key in list(query.obsm.keys()):
        if key.startswith("_scvi"):
            del query.obsm[key]

    if not issparse(query.X):
        query.X = csr_matrix(query.X)

    torch.set_float32_matmul_precision("high")
    scvi.settings.seed = 0
    use_gpu = "${params.scvi_use_gpu}".strip().lower() in {"1", "true", "yes", "y"}
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    scvi.model.SCVI.prepare_query_anndata(query, model_dir)
    vae_q = scvi.model.SCVI.load_query_data(query, model_dir)
    vae_q.is_trained = True
    vae_q.to_device(device)
    latent = vae_q.get_latent_representation(batch_size=${params.batch_size})

    adata_embedding = ad.AnnData(
        X=np.asarray(latent),
        obs=original_obs,
        var=pd.DataFrame(index=[f"V{i+1}" for i in range(latent.shape[1])]),
    )
    if original_spatial is not None:
        adata_embedding.obsm["spatial"] = original_spatial
    adata_embedding.write_h5ad("scvi_embeddings.h5ad")
    """
}


// ============================ batch integration ============================ //
// Native scVI batch integration (the canonical batch_key baseline). Unlike
// `embed_by_scvi` above (zero-shot Census query projection), this trains a fresh
// scVI VAE on the dataset, conditioning the decoder on `batch_id` ONLY. No cell
// type label ever enters the model (that would be scANVI). `cell_type` is carried
// through in obs untouched so the downstream scIB benchmark can score it.

params.batch_key         = "batch_id"
params.scvi_n_latent     = 30
params.scvi_max_epochs   = null   // null -> let scvi-tools auto-pick the epoch count
params.integration_n_hvg = 2000   // 0 -> use all genes (skip HVG subsetting)

process integrate_by_scvi {

    tag "${id}"

    label 'gpu_task'

    container 'scverse/scvi-tools:py3.11-cu12-runtime-stable'

    publishDir "${params.emb_results_dir}/embeddings/scvi_denovo", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("scVI (de novo)"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scvi
    import torch
    from scipy.sparse import csr_matrix, issparse

    adata = ad.read_h5ad("${raw_h5ad}")
    original_obs = adata.obs.copy()
    original_spatial = adata.obsm["spatial"].copy() if "spatial" in adata.obsm else None

    if "${params.batch_key}" not in adata.obs:
        # No batch annotation: train scVI on the dataset as a single batch.
        print("WARNING: obs['${params.batch_key}'] not found; treating all cells as a single batch.")
        adata.obs["${params.batch_key}"] = "batch0"

    # scVI expects raw counts in X.
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)

    # Highly variable gene subsetting (standard for scVI integration). Use scvi's
    # native, batch-aware poisson_gene_selection (the runtime image has no scanpy);
    # fall back to all genes on any failure.
    n_hvg = ${params.integration_n_hvg}
    if n_hvg and adata.n_vars > n_hvg:
        try:
            scvi.data.poisson_gene_selection(
                adata, n_top_genes=n_hvg, batch_key="${params.batch_key}", subset=True,
            )
        except Exception as exc:
            print(f"[integrate_by_scvi] HVG selection skipped ({type(exc).__name__}: {exc}); using all genes.")

    torch.set_float32_matmul_precision("high")
    scvi.settings.seed = 0

    # ONLY batch information is provided to the model.
    scvi.model.SCVI.setup_anndata(adata, batch_key="${params.batch_key}")
    model = scvi.model.SCVI(adata, n_latent=${params.scvi_n_latent}, n_layers=2)
    model.train(max_epochs=${params.scvi_max_epochs ?: 'None'})

    latent = model.get_latent_representation()

    adata_embedding = ad.AnnData(
        X=np.asarray(latent, dtype=np.float32),
        obs=original_obs,
        var=pd.DataFrame(index=[f"V{i+1}" for i in range(latent.shape[1])]),
    )
    if original_spatial is not None:
        adata_embedding.obsm["spatial"] = original_spatial
    adata_embedding.write_h5ad("scvi_integrated_embeddings.h5ad")
    """
}
