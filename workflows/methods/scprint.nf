params.model = "scPRINT/v2-medium.ckpt"
params.batch_size = 64
params.emb_results_dir = "results"
params.how = "random expr"
params.max_len = 4000

process preprocess_for_scprint {

    tag "${id}"

    container "housy17/scprint:latest"

    input:
    tuple val(id), path(raw_h5ad)
    val species

    output:
    tuple val(id), path(raw_h5ad), path("temp_scprint.h5ad")

    script:
    """
    #!/.venv/bin/python
    import scanpy as sc
    adata = sc.read_h5ad("${raw_h5ad}")
    species = "${species}"
    if species.lower() == "human":
        organism_ontology_term_id = "NCBITaxon:9606"

    import pandas as pd
    empty_obs = pd.DataFrame(index=adata.obs_names.tolist())
    empty_obs['barcode'] = adata.obs['barcode'].tolist()
    adata.obs = empty_obs

    adata.obs['organism_ontology_term_id'] = organism_ontology_term_id
    adata.var_names = adata.var['ensembl_id']

    # if "is_primary_data" in adata.obs.columns:
    #     adata.obs.drop(columns="is_primary_data", inplace=True)
    
    adata.write_h5ad("temp_scprint.h5ad")
    """

}

process _embed_by_scprint {

    tag "${id}"

    label "gpu_task"

    container "housy17/scprint:latest"

    input:
    tuple val(id), path(raw_h5ad), path(processed_h5ad)

    output:
    tuple val(id), path(raw_h5ad), path("temp_scprint_embeddings.h5ad")

    script:
    """
    source /.venv/bin/activate
    
    if [ "\$HOME" != "/root" ]; then
        cp -r /root/.lamin "\$HOME"/
    fi
    
    python - << 'EOF'
    import torch
    import os
    import sys
    sys.path.append(os.path.join("/code", "scprint"))
    from scprint import scPrint
    from scdataloader import Preprocessor, utils
    from scdataloader.utils import load_genes
    from scprint.tasks import GNInfer, Embedder, Denoiser, withknn
    
    model_checkpoint_file = os.path.join("/data/model_weights", "${params.model}")

    # make sure that you check if you have a GPU with flashattention or not (see README)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = torch.load(model_checkpoint_file, map_location=device)
    

    transformer = "normal" if device=="cpu" else "flash"

    # both are for compatibility issues with different versions of the pretrained model, so we need to load it with the correct transformer
    if "prenorm" in m["hyper_parameters"]:
        m["hyper_parameters"].pop("prenorm")
        torch.save(m, model_checkpoint_file)
    if "label_counts" in m["hyper_parameters"]:
        # you need to set precpt_gene_emb=None otherwise the model will look for its precomputed gene embeddings files although they were already converted into model weights, so you don't need this file for a pretrained model
        model = scPrint.load_from_checkpoint(
            model_checkpoint_file,
            precpt_gene_emb=None,
            classes=m["hyper_parameters"]["label_counts"],
            transformer=transformer,
        )
    else:
        model = scPrint.load_from_checkpoint(
            model_checkpoint_file, precpt_gene_emb=None, transformer=transformer
        )

    ## There is a bug in scprint, Hou S. updated ##
    ## start ##
    
    for key, value in model.label_decoders['self_reported_ethnicity_ontology_term_id'].items():
        if len(value.split(',')) == 2:
            model.label_decoders['self_reported_ethnicity_ontology_term_id'][key] = value.split(',')[0]

    ##  end  ##

    # this might happen if you have a model that was trained with a different set of genes than the one you are using in the ontology (e.g. newer ontologies), While having genes in the onlogy not in the model is fine. the opposite is not, so we need to remove the genes that are in the model but not in the ontology
    missing = set(model.genes) - set(load_genes(model.organisms).index)
    if len(missing) > 0:
        print(
            "Warning: some genes missmatch exist between model and ontology: solving...",
        )
        model._rm_genes(missing)

    # again if not on GPU you need to convert the model to float64
    if device=="cpu":
        model = model.to(torch.float32)

    # you can perform your inference on float16 if you have a GPU, otherwise use float64
    dtype = torch.float16 if device != "cpu" else torch.float32

    # the models are often loaded with some parts still displayed as "cuda" and some as "cpu", so we need to make sure that the model is fully on the right device
    model = model.to(device)

    embedder = Embedder(
        # can work on random genes or most variables etc..
        how="${params.how}",
        # number of genes to use
        max_len=${params.max_len},
        # the model is trained on a minibatch of 64 cells but you can choose whatever
        batch_size=${params.batch_size},
        # for the dataloading
        num_workers=8,
        # we will only use the cell type embedding here.
        pred_embedding=["cell_type_ontology_term_id"],
        save_every=40_000,
        dtype=dtype,
        doplot=False
    )

    import scanpy as sc
    import pandas as pd
    adata = sc.read_h5ad("${processed_h5ad}")
    # obs_names = adata.obs_names.tolist()

    preprocessor = Preprocessor(
        filter_gene_by_counts = False,
        filter_cell_by_counts = False,
        min_valid_genes_id = 50,
        min_dataset_size = 0,
        min_nnz_genes = 0,
        do_postp=False, skip_validate=True)
    adata = preprocessor(adata)
    adata, metrics = embedder(model, adata, cache=False)

    obs_names = adata.obs["barcode"].tolist()
    # adata.write_h5ad("test.h5ad")

    embedding = adata.obsm['scprint_emb']
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = sc.AnnData(X=embedding, var=pd.DataFrame(index=var_names))
    adata_embedding.obs_names = obs_names
    adata_embedding.write("temp_scprint_embeddings.h5ad")
    EOF
    """
}

process postprocess_for_scprint {

    tag "${id}"
    
    container "housy17/scprint:latest"

    publishDir "${params.emb_results_dir}}/embeddings/scprint", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(processed_h5ad)

    output:
    tuple val(id), val("scPRINT"), path("*_embeddings.h5ad")

    script:
    """
    #!/.venv/bin/python
    import scanpy as sc
    adata = sc.read_h5ad("${raw_h5ad}")
    adata_embedding = sc.read_h5ad("${processed_h5ad}")
    obs = adata.obs.copy()
    obs = obs.loc[adata_embedding.obs_names.tolist()]
    adata_embedding.obs = obs
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata[adata_embedding.obs_names].obsm['spatial']
    adata_embedding.write("scprint_embeddings.h5ad")
    """
}

workflow embed_by_scprint {

    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_scprint(Raw_H5ad_Channel, "human") // 固定只分析人类
        _embed_by_scprint(preprocess_for_scprint.out)
        postprocess_for_scprint(_embed_by_scprint.out)

    emit:
        postprocess_for_scprint.out

}
