params.finetune_label_key = "cell_type"
params.finetune_results_dir = "results"
params.emb_results_dir = ""


process finetune_by_cell_classifier {

    tag "${id}"

    label 'gpu_task'

    container "housy17/scllms:latest"

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models",
               saveAs: { filename -> "${method}/${id}/posthoc_classifier" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), val(method), path(embedding_h5ad)

    output:
    tuple val(id), path("${method}_finetuned_model")

    script:
    """
    python /code/celltype_annotation_finetune.py \\
        --data_path $embedding_h5ad --method_name $method --label_key ${params.finetune_label_key}
    """

}

process predict_by_cell_classifier {

    tag "${id}"

    label 'gpu_task'

    container "housy17/scllms:latest"

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "${method}_predictions.tsv", 
               saveAs: { filename -> "${method}/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), val(method), path(test_embedding_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    python /code/celltype_annotation_predict.py \\
        --method_name $method --data_path $test_embedding_h5ad --model_dir $model_weights
    """

}

process get_prediction_labels {

    tag "${id}"

    container "housy17/scllms:latest"

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy',
               saveAs: { filename -> "${method}/${id}_predicted_labels.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), val(method), path(prediction_tsv)

    output:
    tuple val(id), path("${id}_predicted_labels.tsv")

    script:
    """
    #!/usr/bin/env python

    import pandas as pd
    import numpy as np

    pred_proba = pd.read_csv("${prediction_tsv}", sep="\\t", index_col=0)
    pred_labels = pred_proba.idxmax(axis=1)
    pred_labels_df = pd.DataFrame(pred_labels, columns=['predicted_label'])
    pred_labels_df.to_csv("${id}_predicted_labels.tsv", sep="\\t")
    """

}


//==========================================
//               CELLama
//==========================================

include { _finetune_by_cellama; embed_by_finetuned_cellama } from "../methods/cellama.nf"

workflow finetune_by_cellama {
    take:
        Raw_H5ad_Channel

    main:
        ModelWeight = _finetune_by_cellama(Raw_H5ad_Channel)
        TrainEmbeddings = embed_by_finetuned_cellama(Raw_H5ad_Channel.combine(ModelWeight, by:0))
        finetune_by_cell_classifier(TrainEmbeddings)

    emit:
        finetune_by_cell_classifier.out
            .combine(ModelWeight, by:0)
            .map{ id, classifier_model, embedding_model -> tuple(id, tuple(classifier_model, embedding_model))}

}

workflow predict_by_cellama {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), tuple(path(model_weights), path(embedding_model_weights))
    
    main:
        NormalizedChannel = Raw_H5ad_and_Model_Weights_Channel.map { id, h5ad, models ->
            def model_weights
            def embedding_model_weights

            if (models instanceof List) {
                // tuple
                model_weights           = models[0]
                embedding_model_weights = models[1]
            } else {
                // String or Path
                model_weights           = file(models).resolve("posthoc_classifier")
                embedding_model_weights = file(models).resolve("base_model")
            }

            tuple(id, h5ad, model_weights, embedding_model_weights)
        }

        InputChannel = NormalizedChannel.map{ id, h5ad, mw, emw -> tuple(id, h5ad, emw)}
        embed_by_finetuned_cellama(InputChannel)
        SecondChannel = embed_by_finetuned_cellama.out
            .combine(NormalizedChannel, by: 0)
            .map{ id, method, embedding, foo, mw, emw -> tuple(id, method, embedding, mw)}
        predict_by_cell_classifier(SecondChannel)
        get_prediction_labels(predict_by_cell_classifier.out.map{ id, prob -> tuple(id, "CELLama", prob)})

    emit:
        predict_by_cell_classifier.out
}



//==========================================
//               SCimilarity
//==========================================

include { embed_by_scimilarity } from "../methods/scimilarity.nf"

workflow finetune_by_scimilarity {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_scimilarity(Raw_H5ad_Channel) // id, method_name, embeddings_h5ad
        finetune_by_cell_classifier(InputChannel) // id, method finetuned_model
    emit:
        finetune_by_cell_classifier.out

}

workflow predict_by_scimilarity {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), path(model_weights)
    
    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        embed_by_scimilarity(InputChannel)
        SecondChannel = embed_by_scimilarity.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        predict_by_cell_classifier(SecondChannel)
        get_prediction_labels(predict_by_cell_classifier.out.map{ id, prob -> tuple(id, "SCimilarity", prob)})

    emit:
        predict_by_cell_classifier.out

}


//==========================================
//               GenePT-W
//==========================================

include { embed_by_genept_w } from "../methods/genept.nf"

workflow finetune_by_genept_w {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_genept_w(Raw_H5ad_Channel) // id, method_name, embeddings_h5ad
        finetune_by_cell_classifier(InputChannel) // id, method finetuned_model
    emit:
        finetune_by_cell_classifier.out

}

workflow predict_by_genept_w {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), path(model_weights)
    
    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        embed_by_genept_w(InputChannel)
        SecondChannel = embed_by_genept_w.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        predict_by_cell_classifier(SecondChannel)
        get_prediction_labels(predict_by_cell_classifier.out.map{ id, prob -> tuple(id, "GenePT-w", prob)})

    emit:
        predict_by_cell_classifier.out

}


//==========================================
//               UCE
//==========================================

include { embed_by_uce } from "../methods/uce.nf"

workflow finetune_by_uce {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_uce(Raw_H5ad_Channel) // id, method_name, embeddings_h5ad
        finetune_by_cell_classifier(InputChannel) // id, method finetuned_model
    emit:
        finetune_by_cell_classifier.out
}

workflow predict_by_uce {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), path(model_weights)

    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        embed_by_uce(InputChannel)
        SecondChannel = embed_by_uce.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        predict_by_cell_classifier(SecondChannel)
        get_prediction_labels(predict_by_cell_classifier.out.map{ id, prob -> tuple(id, "UCE", prob)})

    emit:
        predict_by_cell_classifier.out
        

}

//==========================================
//               CellFM
//==========================================

include { finetune_by_cellfm } from "../methods/cellfm.nf"
include { predict_by_cellfm as _predict_by_cellfm } from "../methods/cellfm.nf"

workflow predict_by_cellfm {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), path(model_weights)

    main:
        _predict_by_cellfm(Raw_H5ad_and_Model_Weights_Channel)
        get_prediction_labels(_predict_by_cellfm.out.map{ id, prob -> tuple(id, "CellFM", prob)})

    emit:
        _predict_by_cellfm.out
}


//==========================================
//               CellPLM
//==========================================

include { finetune_by_cellplm } from "../methods/cellplm.nf"
include { predict_by_cellplm as _predict_by_cellplm } from "../methods/cellplm.nf"

workflow predict_by_cellplm {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), path(model_weights)

    main:
        _predict_by_cellplm(Raw_H5ad_and_Model_Weights_Channel)
        get_prediction_labels(_predict_by_cellplm.out.map{ id, prob -> tuple(id, "CellPLM", prob)})

    emit:
        _predict_by_cellplm.out
}

//==========================================
//               scCello
//==========================================

include { tokenize_by_sccello_with_ct as tokenize_by_sccello } from "../methods/sccello.nf"
include { _finetune_by_sccello; _predict_by_sccello } from "../methods/sccello.nf"

workflow finetune_by_sccello {
    take:
        Raw_H5ad_Channel

    main:
        tokenize_by_sccello(Raw_H5ad_Channel, true)
        _finetune_by_sccello(tokenize_by_sccello.out)

    emit:
        _finetune_by_sccello.out
}

workflow predict_by_sccello {
    take:
        Raw_H5ad_and_Model_Weights_Channel

    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        tokenize_by_sccello(InputChannel, false)
        SecondChannel = tokenize_by_sccello.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, raw, processed, foo, model -> tuple(id, processed, model)}
        _predict_by_sccello(SecondChannel)
        get_prediction_labels(_predict_by_sccello.out.map{ id, prob -> tuple(id, "scCello", prob)})
    
    emit:
        _predict_by_sccello.out

}


//==========================================
//               scBERT
//==========================================

include { preprocess_for_scbert } from "../methods/scbert.nf"
include { _finetune_by_scbert; _predict_by_scbert } from "../methods/scbert.nf"

workflow finetune_by_scbert {
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_scbert(Raw_H5ad_Channel)
        _finetune_by_scbert(preprocess_for_scbert.out)

    emit:
        _finetune_by_scbert.out
}

workflow predict_by_scbert {
    take:
        Raw_H5ad_and_Model_Weights_Channel

    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        preprocess_for_scbert(InputChannel)
        SecondChannel = preprocess_for_scbert.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, processed, foo, model -> tuple(id, processed, model)}
        _predict_by_scbert(SecondChannel)
        get_prediction_labels(_predict_by_scbert.out.map{ id, prob -> tuple(id, "scBERT", prob)})
    
    emit:
        _predict_by_scbert.out
}


//==========================================
//               scFoundation
//==========================================

include { preprocess_for_scfoundation } from "../methods/scfoundation.nf"
include { _finetune_by_scfoundation; _predict_by_scfoundation } from "../methods/scfoundation.nf"

workflow finetune_by_scfoundation {
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_scfoundation(Raw_H5ad_Channel)
        _finetune_by_scfoundation(preprocess_for_scfoundation.out)

    emit:
        _finetune_by_scfoundation.out
}

workflow predict_by_scfoundation {
    take:
        Raw_H5ad_and_Model_Weights_Channel

    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        preprocess_for_scfoundation(InputChannel)
        SecondChannel = preprocess_for_scfoundation.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, processed, foo, model -> tuple(id, processed, model)}
        _predict_by_scfoundation(SecondChannel)
        get_prediction_labels(_predict_by_scfoundation.out.map{ id, prob -> tuple(id, "scFoundation", prob)})
    
    emit:
        _predict_by_scfoundation.out
}


//==========================================
//               scPRINT
//==========================================

// scPRINT is officially a ZERO-SHOT annotation model (a hierarchical classifier on the
// cell embedding predicts labels on unseen data; no official finetune recipe). So it is
// evaluated like the other zero-shot methods: frozen embedding + shared linear probe,
// NOT the (off-label, heavy) native scPRINT finetune. The native _finetune_by_scprint /
// _predict_by_scprint processes still exist in methods/scprint.nf but are no longer used.
include { embed_by_scprint } from "../methods/scprint.nf"

workflow finetune_by_scprint {
    take:
        Raw_H5ad_Channel

    main:
        InputChannel = embed_by_scprint(Raw_H5ad_Channel) // id, "scPRINT", embeddings_h5ad
        finetune_by_cell_classifier(InputChannel)

    emit:
        finetune_by_cell_classifier.out
}

workflow predict_by_scprint {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), path(model_weights)

    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        embed_by_scprint(InputChannel)
        SecondChannel = embed_by_scprint.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, method, embedding, foo, model -> tuple(id, method, embedding, model)}
        predict_by_cell_classifier(SecondChannel)
        get_prediction_labels(predict_by_cell_classifier.out.map{ id, prob -> tuple(id, "scPRINT", prob)})

    emit:
        predict_by_cell_classifier.out
}


//==========================================
//               scGPT
//==========================================

include { finetune_by_scgpt } from "../methods/scgpt.nf"
include { predict_by_scgpt as _predict_by_scgpt } from "../methods/scgpt.nf"

workflow predict_by_scgpt {
    take:
        Raw_H5ad_and_Model_Weights_Channel

    main:
        _predict_by_scgpt(Raw_H5ad_and_Model_Weights_Channel)
        get_prediction_labels(_predict_by_scgpt.out.map{ id, prob -> tuple(id, "scGPT", prob)})

    emit:
        _predict_by_scgpt.out
}


//==========================================
//               LangCell
//==========================================

include { tokenize_by_langcell_with_celltype; tokenize_by_langcell } from "../methods/langcell.nf"
include { _finetune_by_langcell; _predict_by_langcell } from "../methods/langcell.nf"

workflow finetune_by_langcell {
    take:
        Raw_H5ad_Channel

    main:
        tokenize_by_langcell_with_celltype(Raw_H5ad_Channel)
        _finetune_by_langcell(tokenize_by_langcell_with_celltype.out)

    emit:
        _finetune_by_langcell.out
}

workflow predict_by_langcell {
    take:
        Raw_H5ad_and_Model_Weights_Channel
        // tuple val(id), path(test_h5ad), path(model_weights)

    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        tokenize_by_langcell(InputChannel)
        SecondChannel = tokenize_by_langcell.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, raw, tokenized, foo, model -> tuple(id, raw, tokenized, model)}
        _predict_by_langcell(SecondChannel)
        get_prediction_labels(_predict_by_langcell.out.map{ id, prob -> tuple(id, "LangCell", prob)})

    emit:
        _predict_by_langcell.out
}


//==========================================
//               Geneformer
//==========================================

include { preprocess_for_geneformer } from "../methods/geneformer.nf"
include { _finetune_by_geneformer; _predict_by_geneformer } from "../methods/geneformer.nf"

workflow finetune_by_geneformer {
    take:
        Raw_H5ad_Channel

    main:
        preprocess_for_geneformer(Raw_H5ad_Channel)
        _finetune_by_geneformer(preprocess_for_geneformer.out)

    emit:
        _finetune_by_geneformer.out
}

workflow predict_by_geneformer {
    take:
        Raw_H5ad_and_Model_Weights_Channel

    main:
        InputChannel = Raw_H5ad_and_Model_Weights_Channel.map{ id, h5ad, model -> tuple(id, h5ad)}
        preprocess_for_geneformer(InputChannel)
        SecondChannel = preprocess_for_geneformer.out
            .combine(Raw_H5ad_and_Model_Weights_Channel, by: 0)
            .map{ id, raw, processed, foo, model -> tuple(id, raw, processed, model)}
        _predict_by_geneformer(SecondChannel)
        get_prediction_labels(_predict_by_geneformer.out.map{ id, prob -> tuple(id, "Geneformer", prob)})
    
    emit:
        _predict_by_geneformer.out
}
