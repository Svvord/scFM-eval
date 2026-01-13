params.label_key = "cell_type"
params.emb_results_dir = ""
params.fewshot_results_dir = "results"

include { embed_by_uce } from "../methods/uce.nf"
include { embed_by_scgpt } from "../methods/scgpt.nf"
include { embed_by_geneformer } from "../methods/geneformer.nf"
include { embed_by_scfoundation } from "../methods/scfoundation.nf"
include { embed_by_cellama } from "../methods/cellama.nf"
include { embed_by_scimilarity } from "../methods/scimilarity.nf"
include { embed_by_scbert } from "../methods/scbert.nf"
include { embed_by_cellfm } from "../methods/cellfm.nf"
include { embed_by_langcell } from "../methods/langcell.nf"
include { embed_by_sccello } from "../methods/sccello.nf"
include { embed_by_scprint } from "../methods/scprint.nf"
include { embed_by_cellplm } from "../methods/cellplm.nf"
include { embed_by_genept_w } from "../methods/genept.nf"


process prototypical_fit {

    tag "${id}"

    container "housy17/scllms:latest"

    publishDir "${params.fewshot_results_dir}/fewshot/fitted_prototypes", mode: 'copy',
               saveAs: { filename -> "${method}/${id}.npz" }, enabled: params.fewshot_results_dir as boolean

    input:
    tuple val(id), val(method), path(embedding_h5ad)

    output:
    tuple val(id), path("${method}_prototypes.npz")

    script:
    """
    #!/usr/bin/env python
    import scanpy as sc
    from scipy.sparse import issparse
    import numpy as np

    adata = sc.read_h5ad("${embedding_h5ad}")
    label_key = "${params.label_key}"
    embeddings = adata.X.toarray() if issparse(adata.X) else adata.X

    prototypes = {}
    for label in adata.obs[label_key].unique():
        prototypes[label] = embeddings[adata.obs[label_key] == label].mean(axis=0)

    np.savez_compressed("${method}_prototypes.npz", **prototypes)
    """
}


process prototypical_inference {

    tag "${id}"

    container "housy17/scllms:latest"

    publishDir "${params.fewshot_results_dir}/fewshot/inference", mode: 'copy', pattern: "${method}_predictions.tsv", 
               saveAs: { filename -> "${method}/${id}_predicted_labels.tsv" }, enabled: params.fewshot_results_dir as boolean
    publishDir "${params.fewshot_results_dir}/fewshot/inference", mode: 'copy', pattern: "${method}_inference.tsv", 
               saveAs: { filename -> "${method}/${id}_predicted_probs.tsv" }, enabled: params.fewshot_results_dir as boolean

    input:
    tuple val(id), val(method), path(query_embedding_h5ad), path(prototypes_npz)

    output:
    tuple val(id), val(method), path("*_inference.tsv"), path("*_predictions.tsv")

    script:
    """
    #!/usr/bin/env python
    import scanpy as sc
    from scipy.sparse import issparse
    import json
    from scipy.spatial.distance import cdist
    import numpy as np
    from scipy.special import softmax
    import pandas as pd

    adata = sc.read_h5ad("${query_embedding_h5ad}")
    barcodes = adata.obs.barcode.tolist()
    query_embeddings = adata.X.toarray() if issparse(adata.X) else adata.X

    loaded = np.load("${prototypes_npz}")
    prototypes = {key: loaded[key] for key in loaded.files}
    
    columns = sorted(list(prototypes.keys()))
    support_embeddings = np.array([prototypes[c] for c in columns])

    dist_matrix = cdist(query_embeddings, support_embeddings, metric='cosine')
    pred_proba = softmax(-dist_matrix, axis=-1)
    pred_proba = pd.DataFrame(pred_proba, columns=columns, index=barcodes)
    pred_proba.to_csv("${method}_inference.tsv", sep="\\t")

    pred_labels = pred_proba.idxmax(axis=1)
    pred_labels_df = pd.DataFrame(pred_labels, columns=['predicted_label'])
    pred_labels_df.to_csv("${method}_predictions.tsv", sep="\\t")
    """

}




//==========================================
//               scGPT
//==========================================

workflow fewshot_by_scgpt {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_scgpt(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}


workflow inference_by_scgpt {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_scgpt(InputChannel)
        SecondChannel = embed_by_scgpt.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}





//==========================================
//               UCE
//==========================================



workflow fewshot_by_uce {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_uce(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}


workflow inference_by_uce {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_uce(InputChannel)
        SecondChannel = embed_by_uce.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}





//==========================================
//               SCimilarity
//==========================================

workflow fewshot_by_scimilarity {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_scimilarity(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}


workflow inference_by_scimilarity {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_scimilarity(InputChannel)
        SecondChannel = embed_by_scimilarity.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}



//==========================================
//               Geneformer
//==========================================


workflow fewshot_by_geneformer {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_geneformer(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_geneformer {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_geneformer(InputChannel)
        SecondChannel = embed_by_geneformer.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}




//==========================================
//               scFoundation
//==========================================


workflow fewshot_by_scfoundation {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_scfoundation(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_scfoundation {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_scfoundation(InputChannel)
        SecondChannel = embed_by_scfoundation.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}


//==========================================
//               CELLama
//==========================================

workflow fewshot_by_cellama {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_cellama(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_cellama {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_cellama(InputChannel)
        SecondChannel = embed_by_cellama.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}




//==========================================
//               scBERT
//==========================================

workflow fewshot_by_scbert {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_scbert(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_scbert {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_scbert(InputChannel)
        SecondChannel = embed_by_scbert.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}


//==========================================
//               CellFM
//==========================================

workflow fewshot_by_cellfm {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_cellfm(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_cellfm {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_cellfm(InputChannel)
        SecondChannel = embed_by_cellfm.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}



//==========================================
//               LangCell
//==========================================


workflow fewshot_by_langcell {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_langcell(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_langcell {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_langcell(InputChannel)
        SecondChannel = embed_by_langcell.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}



//==========================================
//               scCello
//==========================================


workflow fewshot_by_sccello {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_sccello(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_sccello {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_sccello(InputChannel)
        SecondChannel = embed_by_sccello.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}



//==========================================
//               scPRINT
//==========================================

workflow fewshot_by_scprint {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_scprint(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_scprint {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_scprint(InputChannel)
        SecondChannel = embed_by_scprint.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}



//==========================================
//               CellPLM
//==========================================


workflow fewshot_by_cellplm {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_cellplm(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_cellplm {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_cellplm(InputChannel)
        SecondChannel = embed_by_cellplm.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}



//==========================================
//               GenePT-W
//==========================================


workflow fewshot_by_genept_w {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_genept_w(Raw_H5ad_Channel)
        prototypical_fit(InputChannel)

    emit:
        prototypical_fit.out

}

workflow inference_by_genept_w {
    take:
        Raw_H5ad_and_Prototypes_Channel
        // tuple val(id), path(query_h5ad), path(prototypes)

    main:
        InputChannel = Raw_H5ad_and_Prototypes_Channel.map{ id, h5ad, model -> tuple(id, h5ad) }
        embed_by_genept_w(InputChannel)
        SecondChannel = embed_by_genept_w.out
            .combine(Raw_H5ad_and_Prototypes_Channel, by:0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        prototypical_inference(SecondChannel)

    emit:
        prototypical_inference.out
}