params.model = "CELLama/all-MiniLM-L6-v2"
params.emb_results_dir = "results"
params.top_k = 30

process embed_by_cellama {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellama:latest"

    publishDir "${params.emb_results_dir}/embeddings/cellama", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

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



params.finetune_epoch        = 20
// params.finetune_eval_size = 0.1  无效, 但实际上是 0.1
params.finetune_batch_size   = 32   
params.finetune_results_dir  = ""
params.cellama_finetuned_top_k = 30

process _finetune_by_cellama {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellama:latest"

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models", mode: 'copy',
               saveAs: { filename -> "cellama/${id}/base_model" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    #!/usr/bin/env python
    import scanpy as sc
    from cellama_training import sentence_from_adata_generator, train_model_sentences

    adata = sc.read_h5ad("${raw_h5ad}")

    #Training Parameters
    num_samples = 10000
    top_k_values = [16,20,24,28]
    # CELLama obs_features are appended to the input sentence. Target labels are
    # used only by the post-hoc classifier in this pipeline.
    obs_features_options = [None]
    n_hvgs_s=[500,1200]#Already selected samples.

    all_examples= sentence_from_adata_generator.generate_and_save_examples(
        adata, num_samples, top_k_values, obs_features_options,
        n_hvgs_s=n_hvgs_s, 
        genename="gene_symbol",
        save_file = None, 
        verbose=False, return_examples=True
    )

    output_path = "./cellama_finetuned_model"
    train_model_sentences.train_model(
        all_examples, output_path, 
        backcbone_model = "/data/model_weights/${params.model}",
        epochs=${params.finetune_epoch}, 
        batch_size=${params.finetune_batch_size}, 
        evaluation_steps=1000, validation_size=1000,learning_rate=1e-5)
    """

}

process embed_by_finetuned_cellama {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellama:latest"

    input:
    tuple val(id), path(test_h5ad), path(model_weights)

    output:
    tuple val(id), val("CELLama"), path("*_embeddings.h5ad")
    
    """
    #!/usr/bin/env python
    from cellama import lm_cell_embed
    import scanpy as sc
    import numpy as np
    import pandas as pd


    adata = sc.read_h5ad("${test_h5ad}")

    adata_emb = lm_cell_embed(
        adata, top_k=${params.cellama_finetuned_top_k},
        model_name="${model_weights}",
        gene_list=None, obs_features=None,
        return_sentence=False
    )

    var_names = [f'V{i+1}' for i in range(adata_emb.shape[1])]
    adata_emb.var = pd.DataFrame(index=var_names)
    if 'spatial' in adata.obsm:
        adata_emb.obsm['spatial'] = adata.obsm['spatial']
    adata_emb.write("cellama_embeddings.h5ad")
    """
}
