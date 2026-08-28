params.model = "scBERT/panglao_pretrain.pth"
params.batch_size = 64
params.emb_results_dir = "results"
params.bin_num = 5

process preprocess_for_scbert {

    tag "${id}"

    container "housy17/scbert:latest"

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("preproc/temp_scbert.h5ad")

    script:
    """
    #!/usr/bin/env python
    import scanpy as sc
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy import sparse
    import os
    from pathlib import Path

    panglao = sc.read_h5ad("/code/scbert/data/panglao_10000_simplified.h5ad")
    data = sc.read_h5ad("${raw_h5ad}")
    counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
    ref = panglao.var_names.tolist()
    obj = data.var_names.tolist()

    index_list = []
    loc_list = []
    for i in range(len(ref)):
        if ref[i] in obj:
            loc = obj.index(ref[i])
            loc_list.append(loc)
            index_list.append(i)
    counts[:,index_list] = data.X[:,loc_list]
    
    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = panglao.uns

    # sc.pp.filter_cells(new, min_genes=200) 过滤的事情应该由用户提前处理好
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    Path("./preproc").mkdir(exist_ok=True)
    new.write_h5ad("preproc/temp_scbert.h5ad")
    """
}


process _embed_by_scbert {

    tag "${id}"

    label "gpu_task"

    container "housy17/scbert:latest"

    publishDir "${params.emb_results_dir}/embeddings/scbert", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(processed_h5ad)

    output:
    tuple val(id), val("scBERT"), path("*_embeddings.h5ad")


    script:
    """
    python /code/scbert/attn_sum_save_bug_fixed.py \\
        --data_path $processed_h5ad \\
        --original_adata $raw_h5ad \\
        --model_path /data/model_weights/${params.model} \\
        --bin_num $params.bin_num --device gpu --batch_size $params.batch_size
    """
}

workflow embed_by_scbert {
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_scbert(Raw_H5ad_Channel)
        input_data = Raw_H5ad_Channel.combine(preprocess_for_scbert.out, by:0)
        _embed_by_scbert(input_data)

    emit:
        _embed_by_scbert.out
}

params.finetune_epoch        = 100  // official scBERT max epochs; early-stopping (patience 10) governs
params.finetune_eval_size    = 0.2  // 默认就是 0.2
params.finetune_batch_size   = 8
params.predict_batch_size    = 32
params.finetune_results_dir  = ""

process _finetune_by_scbert {

    tag "${id}"
    
    label "gpu_task"

    container "housy17/scbert:latest"

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models",
               saveAs: { filename -> "scBERT/${id}" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(processed_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    def grad_acc = Math.round(3 * 60 / params.finetune_batch_size)
    """
    # --standalone pins the rendezvous to a fixed port (localhost:29400), so two scbert
    # jobs co-located on one node collide (RendezvousConnectionError). Bind a unique free
    # port per run instead -> concurrent per-tissue jobs never clash.
    MASTER_PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); p=s.getsockname()[1]; s.close(); print(p)')
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:\$MASTER_PORT --rdzv-id=scbert_\$MASTER_PORT \\
        --nproc_per_node=1 /code/scbert/finetune_updated.py \\
        --data_path ${processed_h5ad} \\
        --model_path "/data/model_weights/${params.model}" \\
        --batch_size ${params.finetune_batch_size} \\
        --grad_acc ${grad_acc} \\
        --bin_num ${params.bin_num} \\
        --epoch ${params.finetune_epoch}  \\
        --eval_size ${params.finetune_eval_size} \\
        --ckpt_dir "./scbert_finetuned_model/" \\
        --model_name finetune \\
        --label_key ${params.finetune_label_key}
    """
}

process _predict_by_scbert {

    tag "${id}"
    
    label "gpu_task"

    container "housy17/scbert:latest"

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "scBERT/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(processed_test_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    python /code/scbert/predict_updated.py \\
        --data_path ${processed_test_h5ad} \\
        --model_path ${model_weights} \\
        --bin_num ${params.bin_num} \\
        --batch_size ${params.predict_batch_size}
    """

}
