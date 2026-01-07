params.model = "CELLama/all-MiniLM-L6-v2"
params.results_dir = "results"
params.top_k = 30

process embed_by_cellama {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellama:latest"

    publishDir "results/embeddings/cellama", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("CELLama"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    from cellama import lm_cell_embed
    import scanpy as sc
    import numpy as np
    import pandas as pd

    adata = sc.read_h5ad("${raw_h5ad}")

    #adata_emb.X -> sentence embedded results
    adata_emb = lm_cell_embed(
        adata, top_k=${params.top_k}, 
        model_name="/data/model_weights/${params.model}",
        obs_features= None,
        return_sentence=False
    )

    var_names = [f'V{i+1}' for i in range(adata_emb.shape[1])]
    adata_emb.var = pd.DataFrame(index=var_names)
    if 'spatial' in adata.obsm:
        adata_emb.obsm['spatial'] = adata.obsm['spatial']
    adata_emb.write("cellama_embeddings.h5ad")
    """
}
