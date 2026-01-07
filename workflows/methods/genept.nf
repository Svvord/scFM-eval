params.model = "GenePT/ada-002"
params.results_dir = "results"

// params for genept_s
params.ntrunc = 30

process embed_by_genept_w {

    tag "${id}"

    label "cpu_task"

    container "housy17/genept:latest"

    publishDir "${params.results_dir}/embeddings/genept_w", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("GenePT-w"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import os
    import scanpy as sc
    import pickle
    from scipy.sparse import issparse
    import numpy as np
    import anndata
    import pandas as pd

    adata = sc.read_h5ad("${raw_h5ad}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if "${params.model}" == "GenePT/ada-002":
        weights_file = "GenePT/GenePT_gene_embedding_ada_text.pickle"
        EMBED_DIM = 1536 # embedding dim from ada-002
    elif "${params.model}" == "GenePT/model-3-large":
        weights_file = "GenePT/GenePT_gene_protein_embedding_model_3_text.pickle"
        EMBED_DIM = 3072 # embedding dim from model-3-large
    with open(os.path.join("/data/model_weights", weights_file), "rb") as fp:
        gene_embeddings = pickle.load(fp)
    
    gene_names= adata.var_names.tolist()
    count_missing = 0
    
    lookup_embed = np.zeros(shape=(len(gene_names),EMBED_DIM))

    for i, gene in enumerate(gene_names):
        if gene in gene_embeddings:
            lookup_embed[i,:] = np.array(gene_embeddings[gene]).flatten()
        else:
            count_missing+=1

    if issparse(adata.X):
        genePT_w_emebed = np.dot(adata.X.toarray(),lookup_embed)/len(gene_names)
    else:
        genePT_w_emebed = np.dot(adata.X,lookup_embed)/len(gene_names)
    
    print(f"Unable to match {count_missing} out of {len(gene_names)} genes in the GenePT-w embedding")

    embedding = genePT_w_emebed
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = anndata.AnnData(X=embedding, obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write("genept-w_embeddings.h5ad")
    """

    
}

process embed_by_genept_s {

    tag "${id}"

    container "housy17/genept:latest"

    publishDir "${params.results_dir}/embeddings/genept_s", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("GenePT-s"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import os
    from dotenv import load_dotenv
    load_dotenv(os.path.join("/code", ".env"))
    import numpy as np
    from openai import OpenAI
    import scanpy as sc
    import scipy.sparse as sp
    import anndata
    import pandas as pd

    print(os.getenv("OPENAI_API_KEY"))

    client = OpenAI()
    def get_seq_embed_gpt(X, gene_names, prompt_prefix="", trunc_index = None):
        n_genes = X.shape[1]
        if trunc_index is not None and not isinstance(trunc_index, int):
            raise Exception('trunc_index must be None or an integer!')
        elif isinstance(trunc_index, int) and trunc_index>=n_genes:
            raise Exception('trunc_index must be smaller than the number of genes in the dataset')
        get_test_array = []
        for cell in (X):
            zero_indices = (np.where(cell==0)[0])
            gene_indices = np.argsort(cell)[::-1]
            filtered_genes = gene_indices[~np.isin(gene_indices, list(zero_indices))]
            if trunc_index is not None:
                get_test_array.append(np.array(gene_names[filtered_genes])[0:trunc_index]) 
            else:
                get_test_array.append(np.array(gene_names[filtered_genes])) 
        get_test_array_seq = [prompt_prefix+' '.join(x) for x in get_test_array]
        return(get_test_array_seq)

    def get_gpt_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\\n", " ")
        return np.array(client.embeddings.create(input = text, model=model).data[0].embedding)

    def get_dense_X(adata):
        X = adata.X
        if sp.issparse(X):
            return X.toarray()
        else:
            return np.array(X)

    
    adata = sc.read_h5ad("${raw_h5ad}")
    N_TRUNC_GENE = min(${params.ntrunc}, adata.shape[1])
    cells_data = get_seq_embed_gpt(
        get_dense_X(adata),
        np.array(adata.var_names.tolist()), 
        prompt_prefix = 'A cell with genes ranked by expression: ',
        trunc_index=N_TRUNC_GENE
    )

    embedding = []
    for x in cells_data:
        embedding.append(get_gpt_embedding(x))
    embedding = np.array(embedding)
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = anndata.AnnData(X=embedding, obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write("genept_embeddings.h5ad")
    """
}