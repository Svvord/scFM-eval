params.model = "UCE/33l_8ep_1024t_1280.torch"
params.batch_size = 64
params.results_dir = "results"
params.nlayers = 33

process _embed_by_uce {

    tag "${id}"

    label "gpu_task"

    container 'housy17/uce:latest'

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("results/*_uce_adata.h5ad")

    script:
    """
    mkdir ./results
    python /code/uce/eval_single_anndata.py \\
        --adata_path $raw_h5ad --dir ./results/ \\
        --model_loc "/data/model_weights/$params.model" \\
        --spec_chrom_csv_path "/data/model_weights/UCE/species_chrom.csv" \\
        --token_file "/data/model_weights/UCE/all_tokens.torch" \\
        --protein_embeddings_dir "/data/model_weights/UCE/protein_embeddings" \\
        --offset_pkl_path "/data/model_weights/UCE/species_offsets.pkl" \\
        --nlayers $params.nlayers \\
        --batch_size $params.batch_size
    """
}

process postprocess_for_uce {

    tag "${id}"

    container 'housy17/uce:latest'

    publishDir "${params.results_dir}/embeddings/uce", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }

    input:
    tuple val(id), path(uce_h5ad)

    output:
    tuple val(id), val("UCE"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import scanpy as sc
    import anndata
    import pandas as pd

    adata = sc.read_h5ad("${uce_h5ad}")
    embedding = adata.obsm['X_uce']
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = anndata.AnnData(X=embedding, obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write("uce_embeddings.h5ad")
    """
}

workflow embed_by_uce {
    take:
        Raw_H5ad_Channel

    main:
        _embed_by_uce(Raw_H5ad_Channel)
        postprocess_for_uce(_embed_by_uce.out)
    emit:
        postprocess_for_uce.out
}

