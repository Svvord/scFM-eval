params.model = "scFoundation/models.ckpt"
params.emb_results_dir = "results"
params.scfoundation_pool_type = "all"

process preprocess_for_scfoundation {

    tag "${id}"

    container 'housy17/scfoundation:latest'

    input:
    tuple val(id), path(raw_h5ad) //raw count matrix

    output:
    tuple val(id), path("temp_scfoundation.h5ad") //preprocessed data

    script:
    """
    #!/usr/bin/env python
    import sys
    import os
    from pathlib import Path
    from scipy.sparse import issparse
    sys.path.append('/code/scfoundation')
    from scRNA_workflow import *

    adata = sc.read_h5ad("${raw_h5ad}")

    if issparse(adata.X):
        X_df= pd.DataFrame(adata.X.toarray(),index=adata.obs.index.tolist(),columns=adata.var.index.tolist())
    else:
        X_df = pd.DataFrame(adata.X, index=adata.obs.index.tolist(), columns=adata.var.index.tolist())

    gene_list_df = pd.read_csv('/code/scfoundation/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\\t')
    gene_list = list(gene_list_df['gene_name'])
    X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
    adata_uni = sc.AnnData(X_df)
    adata_uni.obs = adata.obs
    adata_uni.uns = adata.uns
    if 'spatial' in adata.obsm:
        adata_uni.obsm['spatial'] = adata.obsm['spatial']
    adata_uni = BasicFilter(adata_uni,qc_min_genes=0,qc_min_cells=0) # filter cell and gene by lower limit
    adata_uni = QC_Metrics_info(adata_uni)

    save_adata_h5ad(adata_uni,f"./temp_scfoundation.h5ad")
    """
    
}

process _embed_by_scfoundation {

    tag "${id}"
    // embedding for each cell
    label 'gpu_task'

    container 'housy17/scfoundation:latest'

    input:
    tuple val(id), path(processed_h5ad)

    output:
    tuple val(id), path("*.npy")

    script:
    """
    python /code/scfoundation/get_embedding.py \\
    --input_type singlecell \\
    --output_type cell \\
    --pool_type ${params.scfoundation_pool_type} \\
    --tgthighres t4 \\
    --data_path ${processed_h5ad} \\
    --save_path "./" \\
    --pre_normalized F \\
    --version ce \\
    --model_path /data/model_weights/${params.model} \\
    --ckpt_name 01B-resolution
    """
    
}

process postprocess_for_scfoundation {

    tag "${id}"

    container 'housy17/scfoundation:latest'

    publishDir "${params.emb_results_dir}/embeddings/scfoundation", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(embedding_npy)

    output:
    tuple val(id), val("scFoundation"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python

    import anndata as ad
    import numpy as np
    import pandas as pd

    ori_adata = ad.read_h5ad("${raw_h5ad}", backed='r')
    obs = ori_adata.obs.copy()  # 提取 obs DataFrame
    
    embedding = np.load("${embedding_npy}")
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = ad.AnnData(X=embedding, obs=obs, var=pd.DataFrame(index=var_names))
    if 'spatial' in ori_adata.obsm:
        adata_embedding.obsm['spatial'] = ori_adata.obsm['spatial']
    adata_embedding.write("scfoundation_embeddings.h5ad")
    """
}


workflow embed_by_scfoundation{
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_scfoundation(Raw_H5ad_Channel)
        _embed_by_scfoundation(preprocess_for_scfoundation.out)
        postprocess_input = Raw_H5ad_Channel.combine(_embed_by_scfoundation.out, by:0)
        postprocess_for_scfoundation(postprocess_input)

    emit:
        postprocess_for_scfoundation.out
}

params.finetune_epoch        = 20
params.finetune_eval_size    = 0.2
params.finetune_batch_size   = 32
params.predict_batch_size    = 64
params.finetune_results_dir  = ""
// training-strategy knobs (A/B): original = lr 5e-5 + scheduler 'none'
params.scfoundation_lr        = 1e-4
params.scfoundation_scheduler = "warmup_cosine"
params.scfoundation_grad_clip = 0.0

process _finetune_by_scfoundation {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/scfoundation:latest'

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models",
               saveAs: { filename -> "scFoundation/${id}" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(processed_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    python /code/scfoundation/finetune_model_updated.py \\
    --data_path ${processed_h5ad} \\
    --ckpt_path "/data/model_weights/${params.model}" \\
    --epochs ${params.finetune_epoch} \\
    --batch_size ${params.finetune_batch_size} \\
    --val_size ${params.finetune_eval_size} \\
    --label_key ${params.finetune_label_key} \\
    --lr ${params.scfoundation_lr} \\
    --scheduler ${params.scfoundation_scheduler} \\
    --grad_clip ${params.scfoundation_grad_clip}
    """
}

process _predict_by_scfoundation {

    tag "${id}"

    label "gpu_task"

    container 'housy17/scfoundation:latest'

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "scFoundation/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(processed_test_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    python /code/scfoundation/predict_updated.py \\
    --data_path ${processed_test_h5ad} \\
    --ckpt_path ${model_weights} \\
    --backbone_path "/data/model_weights/${params.model}" \\
    --batch_size ${params.predict_batch_size}
    """

}
