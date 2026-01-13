params.model = "scGPT/scGPT_human"
params.batch_size = 64
params.emb_results_dir = "results"

process embed_by_scgpt {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/scgpt:0.2.4'

    publishDir "${params.emb_results_dir}/embeddings/scgpt", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad) // update: raw_count, data transformed codes inside
    //  log1p-transformed + hgnc symbol

    output:
    tuple val(id), val("scGPT"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import scgpt as scg
    import scanpy as sc
    from pathlib import Path
    import os
    import pandas as pd

    adata = sc.read("${raw_h5ad}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    gene_col = "index"
    model_dir = os.path.join("/data/model_weights", "${params.model}")

    adata_embedding = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col = gene_col,
        batch_size = ${params.batch_size},
        return_new_adata = False,
    )
    
    embedding = adata_embedding.obsm['X_scGPT']
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = sc.AnnData(X=embedding, obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write(f"scgpt_embeddings.h5ad")

    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    """
}