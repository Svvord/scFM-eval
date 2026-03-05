params.model = "C2S/C2S-Pythia-410m-cell-type-prediction"
params.batch_size = 8
params.emb_results_dir = "results"

process embed_by_c2s {

    tag "${id}"

    label "gpu_task"

    container 'housy17/c2s:latest'

    publishDir "${params.emb_results_dir}/embeddings/c2s", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("C2S"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    # Python built-in libraries
    import os
    import random
    from collections import Counter

    # Third-party libraries
    import numpy as np
    from tqdm import tqdm

    # Single-cell libraries
    import anndata
    import scanpy as sc

    import argparse

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)

    # Cell2Sentence imports
    import cell2sentence as cs
    from cell2sentence.tasks import embed_cells

    adata = sc.read_h5ad("${raw_h5ad}")
    obs = adata.obs.copy()

    # c2s-specific 
    adata.var['gene_name'] = adata.var['gene_symbol'].tolist()
    adata.obs['organism'] = "Homo sapiens"
    # c2s requires the cell_type column to be present; otherwise, it will raise an error.
    adata.obs['cell_type'] = "NAN"

    # filter should be conducted by user
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)

    # Count normalization
    sc.pp.normalize_total(adata)
    # Lop1p transformation with base 10 - base 10 is important for C2S transformation!
    sc.pp.log1p(adata, base=10)  

    adata_obs_cols_to_keep = ['barcode', 'organism', 'cell_type']
    # Create CSData object
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata, 
        random_state=SEED, 
        sentence_delimiter=' ',
        label_col_names=adata_obs_cols_to_keep
    )

    os.mkdir("./temp/")
    c2s_save_dir = "./temp/"  # C2S dataset will be saved into this directory
    c2s_save_name = "c2s_dataset" 

    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds, 
        vocabulary=vocabulary,
        save_dir=c2s_save_dir,
        save_name=c2s_save_name,
        dataset_backend="arrow"
    )

    # Define CSModel object
    cell_type_prediction_model_path = os.path.join("/data/model_weights", "${params.model}")
    save_dir = "./temp_model/"
    save_name = "cell_embedding_prediction"
    csmodel = cs.CSModel(
        model_name_or_path=cell_type_prediction_model_path,
        save_dir=save_dir,
        save_name=save_name
    )

    embedding = embed_cells(
        csdata=csdata,
        csmodel=csmodel,
        n_genes=200,
        inference_batch_size=${params.batch_size},
    )

    import pandas as pd
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = sc.AnnData(X=embedding, obs=obs, var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write(f"c2s_embeddings.h5ad")
    """
}
