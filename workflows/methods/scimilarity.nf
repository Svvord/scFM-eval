params.model = "SCimilarity/model_v1.1"
params.emb_results_dir = "results"

process preprocess_for_scimilarity {

    tag "${id}"

    container "housy17/scimilarity:latest"

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("temp_scimilarity.h5ad")

    script:
    """
    #!/usr/bin/env python3
    import scanpy as sc
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix

    adata = sc.read_h5ad("${raw_h5ad}")

    if "counts" in adata.layers:
        if isinstance(adata.layers["counts"], np.ndarray):
            adata.layers["counts"] = csr_matrix(adata.layers["counts"])
    else:
        if isinstance(adata.X, np.ndarray):
            adata.layers["counts"] = csr_matrix(adata.X)
        else:
            adata.layers["counts"] = adata.X.copy()
    adata.X = csr_matrix(adata.layers["counts"].shape)
    adata.write_h5ad("temp_scimilarity.h5ad")
    """
}

process _embed_by_scimilarity {

    tag "${id}"

    label "cpu_task"  // scimilarity 不使用 GPU

    container "housy17/scimilarity:latest"

    publishDir "${params.emb_results_dir}/embeddings/scimilarity", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("SCimilarity"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python3
    import scanpy as sc
    from scimilarity.utils import lognorm_counts, align_dataset
    from scimilarity import CellQuery
    from pathlib import Path
    import os
    model_path = "/data/model_weights/${params.model}"
    cq = CellQuery(model_path)

    adams = sc.read("${raw_h5ad}")
    adams = align_dataset(adams, cq.gene_order)
    adams = lognorm_counts(adams)

    embeddings = cq.get_embeddings(adams.X)

    var_names = [f'V{i+1}' for i in range(embeddings.shape[1])]
    import pandas as pd
    adata_emb = sc.AnnData(X=embeddings, obs=adams.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adams.obsm:
        adata_emb.obsm['spatial'] = adams.obsm['spatial']
    adata_emb.write(f"scimilarity_embeddings.h5ad")
    """
}

workflow embed_by_scimilarity {
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_scimilarity(Raw_H5ad_Channel)
        _embed_by_scimilarity(preprocess_for_scimilarity.out)

    emit:
        _embed_by_scimilarity.out
}