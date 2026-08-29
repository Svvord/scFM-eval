params.model = "scCello/scCello-zeroshot"
params.batch_size = 64
params.emb_results_dir = "results"

process tokenize_by_sccello {

    tag "${id}"

    container "housy17/sccello:latest"

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path(raw_h5ad), path("results/tokenized_dataset") // tokenized dataset

    script:
    """
    python /code/sccello/sccello/script/run_data_transformation_updated.py \\
        --h5ad_data_path ${raw_h5ad}
    """
}

process _embed_by_sccello {

    tag "${id}"

    label "gpu_task"

    container "housy17/sccello:latest"

    input:
    tuple val(id), path(raw_h5ad), path(tokenized_dataset)

    output:
    tuple val(id), path(raw_h5ad), path("temp_sccello.h5ad")

    script:
    """
    #!/usr/bin/env python
    import sys
    sys.path.append("/code/sccello")
    from sccello.src.model_prototype_contrastive import PrototypeContrastiveForMaskedLM
    from datasets import load_from_disk
    import torch
    from tqdm import tqdm

    device = "gpu"
    if device != "cpu":
        assert torch.cuda.is_available(), "GPU is not available."
    if device == "gpu":
        device = "cuda"

    dataset = load_from_disk("${tokenized_dataset}")
    obs_names = dataset['barcode']

    remove_columns = dataset.column_names
    remove_columns.remove('input_ids')
    dataset = dataset.remove_columns(remove_columns)

    from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification
    from torch.utils.data import DataLoader
    collator = DataCollatorForCellClassification(model_input_name=["input_ids"])
    batchsize = ${params.batch_size}
    dataloader = DataLoader(dataset, batch_size=batchsize, collate_fn=collator, shuffle=False)
    
    embedding = torch.zeros(len(dataset), 256)
    model = PrototypeContrastiveForMaskedLM.from_pretrained("/data/model_weights/${params.model}", output_hidden_states=True)
    model.eval()
    model.to(device)

    with torch.inference_mode():
        for i, d in tqdm(enumerate(dataloader)):
            cellemb = model(d['input_ids'].to(device), None, d['attention_mask'].to(device), output_hidden_states=True)['logits'][:,-1,:]
            embedding[i * batchsize: (i + 1) * batchsize] = cellemb.cpu()

    embedding = embedding.numpy()
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    import anndata
    import pandas as pd
    adata_embedding = anndata.AnnData(X=embedding, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
    adata_embedding.write("temp_sccello.h5ad")
    """
}


process postprocess_for_sccello {

    tag "${id}"

    container "housy17/sccello:latest"

    publishDir "${params.emb_results_dir}/embeddings/sccello", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(embedding)

    output:
    tuple val(id), val("scCello"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python

    import anndata as ad

    ori_adata = ad.read_h5ad("${raw_h5ad}", backed='r')
    obs = ori_adata.obs.copy()  # 提取 obs DataFrame
    
    adata_embedding = ad.read_h5ad("${embedding}")
    obs = obs.loc[adata_embedding.obs_names]
    adata_embedding.obs = obs
    if 'spatial' in ori_adata.obsm:
        adata_embedding.obsm['spatial'] = ori_adata[adata_embedding.obs_names].obsm['spatial']
    adata_embedding.write("sccello_embeddings.h5ad")
    """
    
}

workflow embed_by_sccello {
    take:
        RawInputChannel

    main:
        tokenize_by_sccello(RawInputChannel)
        _embed_by_sccello(tokenize_by_sccello.out)
        postprocess_for_sccello(_embed_by_sccello.out)

    emit:
        postprocess_for_sccello.out
}


params.finetune_epoch        = 20
params.finetune_eval_size    = 0.1
params.finetune_batch_size   = 24   // official scCello probing batch size
params.predict_batch_size    = 64
params.finetune_results_dir  = ""

process tokenize_by_sccello_with_ct {

    tag "${id}"

    container "housy17/sccello:latest"

    input:
    tuple val(id), path(raw_h5ad)
    val add_cell_type  // 添加布尔值参数

    output:
    tuple val(id), path(raw_h5ad), path("results/tokenized_dataset") // tokenized dataset

    script:
    def cell_type_flag = add_cell_type ? '--add_cell_type' : ''
    """
    python /code/sccello/sccello/script/run_data_transformation_updated.py \\
        --h5ad_data_path ${raw_h5ad} --label_key ${params.finetune_label_key} ${cell_type_flag}
    """
}

process _finetune_by_sccello {

    tag "${id}"

    label "gpu_task"

    container "housy17/sccello:latest"

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models", mode: 'copy',
               saveAs: { filename -> "sccello/${id}" }, enabled: params.finetune_results_dir as boolean
    
    input:
    tuple val(id), path(raw_h5ad), path(tokenized_dataset)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    torchrun --standalone --nproc_per_node=1 /code/sccello/finetune.py \\
        --pretrained_model "/data/model_weights/${params.model}" \\
        --tokenized_dataset ${tokenized_dataset} \\
        --batch_size ${params.finetune_batch_size} \\
        --eval_size ${params.finetune_eval_size} \\
        --epochs ${params.finetune_epoch}
    """
    
}


process _predict_by_sccello {

    tag "${id}"

    label "gpu_task"

    container "housy17/sccello:latest"

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "sccello/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(processed_test_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    #!/usr/bin/env python
    import os
    import sys
    sys.path.append("/code/sccello")
    import json
    from pathlib import Path
    from datasets import load_from_disk
    from sccello.src.model_prototype_contrastive import PrototypeContrastiveForSequenceClassification
    from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification
    from transformers import TrainingArguments, Trainer
    import torch
    from scipy.special import softmax

    dataset = load_from_disk("${processed_test_h5ad}")
    model_dir = Path("${model_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrototypeContrastiveForSequenceClassification.from_pretrained(model_dir).to(device)

    args = TrainingArguments(
        output_dir="test_output",
        per_device_eval_batch_size=${params.predict_batch_size},
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorForCellClassification(
            model_input_name=["input_ids"],
            pad_to_multiple_of=None
        ),
    )

    pred = trainer.predict(dataset)
    probs = softmax(pred.predictions, axis=-1)
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    import pandas as pd
    df = pd.DataFrame(probs, columns=label_names, index=dataset['barcode'])
    df.to_csv("sccello_predictions.tsv", sep="\\t")
    """

}
