params.model = "CellPLM/20231027_85M.best.ckpt"
params.results_dir = "results"

process embed_by_cellplm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellplm:latest"

    publishDir "${params.results_dir}/embeddings/cellplm", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }

    input:
    tuple val(id), path(raw_h5ad) // 内部转成 log1p transformed

    output:
    tuple val(id), val("CellPLM"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import warnings
    warnings.filterwarnings("ignore")
    
    import hdf5plugin
    import numpy as np
    import anndata as ad
    from scipy.sparse import csr_matrix
    from CellPLM.utils import set_seed
    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
    import scanpy as sc
    import matplotlib.pyplot as plt
    import rapids_singlecell as rsc
    import os
    import torch
    import anndata
    import pandas as pd


    PRETRAIN_VERSION = os.path.basename("${params.model}").split('.')[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad("${raw_h5ad}")
    adata.var_names = adata.var['ensembl_id'].tolist()
    adata.var_names_make_unique()

    PRETRAIN_DIR = os.path.join("/data/model_weights", "CellPLM")
    pipeline = CellEmbeddingPipeline(
        pretrain_prefix=PRETRAIN_VERSION,
        pretrain_directory=PRETRAIN_DIR
    )
    embedding = pipeline.predict(adata, # An AnnData object
                device=device)
    embedding = embedding.detach().cpu().numpy()
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = anndata.AnnData(X=embedding, obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write("cellplm_embeddings.h5ad")
    """
    
}