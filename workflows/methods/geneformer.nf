params.model = "Geneformer/Geneformer-V2-316M"
params.batch_size = 64
params.emb_results_dir = "results"

process preprocess_for_geneformer {

    tag "${id}"

    container 'housy17/geneformer:latest'

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path(raw_h5ad), path("preproc")

    script:
    """
    #!/usr/bin/env python
    import scanpy as sc
    from pathlib import Path
    from scipy.sparse import issparse, csr_matrix

    adata = sc.read_h5ad("${raw_h5ad}")
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.obs['barcode'] = adata.obs_names.tolist()
    # ensembl_id 应在数据预处理的步骤加入, 
    # 不在 geneformer 的预处理加入是因为有好几个方法都使用 ensembl
    # adata.var['ensembl_id'] = xxx
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    Path("./preproc").mkdir(exist_ok=True)
    adata.write_h5ad("preproc/temp_geneformer.h5ad")
    """
}

process _embed_by_geneformer {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/geneformer:latest'

    input:
    tuple val(id), path(raw_h5ad), path(processed_raw_h5ad)

    output:
    tuple val(id), path(raw_h5ad), path("embeddings.csv")

    script:
    """
    #!/usr/bin/env python
    import os
    from geneformer import EmbExtractor
    from geneformer import TranscriptomeTokenizer

    tk = TranscriptomeTokenizer({'barcode': 'barcode'}, nproc=16)  # for V1 model, set model_version="V1"
    tk.tokenize_data(
        './preproc',            # "data_directory"
        './',                   # "output_directory"
        "transformed",          # "output_prefix"
        file_format="h5ad"
    )

    model_version = "${params.model}".split('-')[1]
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_mode='cell',
        model_version=model_version,
        max_ncells=None,
        emb_label=['barcode'],
        nproc=16,
        forward_batch_size=${params.batch_size}
    )


    emb = embex.extract_embs(
        os.path.join('/data/model_weights', '${params.model}'),
        'transformed.dataset', './', 'embeddings')
    emb.index = emb['barcode'].values
    """
}

process postprocess_for_geneformer {

    tag "${id}"

    container 'housy17/geneformer:latest'

    publishDir "${params.emb_results_dir}/embeddings/geneformer", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(embedding)

    output:
    tuple val(id), val("Geneformer"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python

    import anndata as ad
    import pandas as pd

    ori_adata = ad.read_h5ad("${raw_h5ad}", backed='r')
    obs = ori_adata.obs.copy()  # 提取 obs DataFrame
    
    embedding = pd.read_csv("${embedding}", index_col=0)
    embedding.index = [str(item) for item in embedding['barcode'].values]
    embedding.drop(columns=['barcode'], inplace=True, errors='ignore')
    embedding = embedding.loc[ori_adata.obs_names, :].values

    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = ad.AnnData(X=embedding, obs=obs, var=pd.DataFrame(index=var_names))
    if 'spatial' in ori_adata.obsm:
        adata_embedding.obsm['spatial'] = ori_adata.obsm['spatial']
    adata_embedding.write("geneformer_embeddings.h5ad")
    """
    
}

workflow embed_by_geneformer {
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_geneformer(Raw_H5ad_Channel)
        _embed_by_geneformer(preprocess_for_geneformer.out)
        postprocess_for_geneformer(_embed_by_geneformer.out)

    emit:
        postprocess_for_geneformer.out
}
