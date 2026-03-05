params.batch_size = 64
params.emb_results_dir = "results"

process tokenize_by_langcell {

    tag "${id}"

    container "housy17/langcell:latest"

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path(raw_h5ad), path("tokenized_dataset")

    script:
    """
    #!/usr/bin/env python
    import sys
    import os
    sys.path.append("/code/langcell/data_preprocess")
    from utils import LangCellTranscriptomeTokenizer
    import scanpy as sc
    import pandas as pd

    data = sc.read_h5ad("${raw_h5ad}")
    
    empty_obs = pd.DataFrame(index=data.obs_names.tolist())
    empty_obs['barcode'] = data.obs['barcode'].tolist()
    data.obs = empty_obs
    data.obs['n_counts'] = data.X.sum(axis=1)

    # ensembl_id 应在数据预处理的步骤加入
    # data.var['ensembl_id'] = data.var['feature_id']

    tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in data.obs.keys()]), nproc=4)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(data)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    tokenized_dataset.save_to_disk('tokenized_dataset')
    """

}

process _embed_by_langcell {

    tag "${id}"

    label "gpu_task"

    container "housy17/langcell:latest"

    input:
    tuple val(id), path(raw_h5ad), path(tokenized_dataset)

    output:
    tuple val(id), path(raw_h5ad), path("temp_langcell.h5ad")

    script:
    """
    #!/usr/bin/env python
    import os
    from datasets import load_from_disk
    import torch.nn as nn, torch.nn.functional as F
    import torch, json
    from transformers import BertTokenizer, BertModel
    import sys
    sys.path.append(os.path.join("/code", "langcell","LangCell-annotation-zeroshot"))
    from utils import BertModel as MedBertModel
    from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    device = "gpu"
    if device != "cpu":
        assert torch.cuda.is_available(), "GPU is not available."
    if device == "gpu":
        device = "cuda"

    class Pooler(nn.Module):
        def __init__(self, config, pretrained_proj, proj_dim):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, proj_dim)
            self.proj.load_state_dict(torch.load(pretrained_proj))
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            pooled_output = hidden_states[:, 0]
            pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
            return pooled_output
    
    MODEL_WEIGHTS_DIR = os.path.join('/data/model_weights', 'LangCell/ckpt')
    CELL_BERT_DIR = os.path.join(MODEL_WEIGHTS_DIR, 'cell_bert')
    CELL_PROJ_FILE = os.path.join(MODEL_WEIGHTS_DIR, 'cell_proj.bin')
    TEXT_BERT_DIR = os.path.join(MODEL_WEIGHTS_DIR, 'text_bert')
    TEXT_PROJ_FILE = os.path.join(MODEL_WEIGHTS_DIR, 'text_proj.bin')
    CMD_HEAD_FILE = os.path.join(MODEL_WEIGHTS_DIR, 'ctm_head.bin')

    model = BertModel.from_pretrained(CELL_BERT_DIR)
    model.pooler = Pooler(model.config, pretrained_proj=CELL_PROJ_FILE, proj_dim=256)
    proj = model.pooler.proj
    model = model.to(device)

    text_encoder = MedBertModel.from_pretrained(TEXT_BERT_DIR, add_pooling_layer=True)
    text_encoder.pooler = Pooler(text_encoder.config, pretrained_proj=TEXT_PROJ_FILE, proj_dim=256)
    text_encoder = text_encoder.to(device)


    ctm_head = nn.Linear(text_encoder.config.hidden_size, 2)
    ctm_head.load_state_dict(torch.load(CMD_HEAD_FILE))
    ctm_head = ctm_head.to(device)

    def cell_encode(cell_input_ids, cell_atts):
        cell = model(cell_input_ids.to(device), cell_atts.to(device))
        cell_last_h = cell.last_hidden_state
        cell_pooler = cell.pooler_output
        return cell_last_h, cell_pooler


    dataset = load_from_disk("${tokenized_dataset}")
    obs_names = dataset['barcode']
    remove_columns = dataset.column_names
    remove_columns.remove('input_ids')
    dataset = dataset.remove_columns(remove_columns)

    batchsize = ${params.batch_size}
    collator = DataCollatorForCellClassification()
    dataloader = DataLoader(dataset, batch_size=batchsize, collate_fn=collator, shuffle=False)

    embedding = torch.zeros(len(dataset), 256)
    model.eval()
    text_encoder.eval()
    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader)):
            cell_last_h, cellemb = cell_encode(d['input_ids'], d['attention_mask']) # batchsize * 256
            embedding[i * batchsize: (i + 1) * batchsize] = cellemb.cpu()

    embedding = embedding.numpy()
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    import anndata
    import pandas as pd
    adata_embedding = anndata.AnnData(X=embedding, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
    adata_embedding.write("temp_langcell.h5ad")
    """
}

process postprocess_for_langcell {

    tag "${id}"

    container "housy17/langcell:latest"

    publishDir "${params.emb_results_dir}/embeddings/langcell", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(embedding)

    output:
    tuple val(id), val("LangCell"), path("*embeddings.h5ad")

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
    adata_embedding.write("langcell_embeddings.h5ad")
    """
}

workflow embed_by_langcell {
    take:
        Raw_H5ad_Channel

    main:
        tokenize_by_langcell(Raw_H5ad_Channel)
        _embed_by_langcell(tokenize_by_langcell.out)
        postprocess_for_langcell(_embed_by_langcell.out)
        
    emit:
        postprocess_for_langcell.out
}


params.finetune_batch_size = 4
params.finetune_epoch = 4
params.finetune_eval_size = 0.33
params.predict_batch_size = 16
params.finetune_results_dir  = ""

process tokenize_by_langcell_with_celltype {

    tag "${id}"

    container "housy17/langcell:latest"

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path(raw_h5ad), path("tokenized_dataset")

    script:
    """
    #!/usr/bin/env python
    import sys
    import os
    sys.path.append("/code/langcell/data_preprocess")
    from utils import LangCellTranscriptomeTokenizer
    import scanpy as sc
    import pandas as pd

    data = sc.read_h5ad("${raw_h5ad}")
    
    empty_obs = pd.DataFrame(index=data.obs_names.tolist())
    empty_obs['barcode'] = data.obs['barcode'].tolist()
    empty_obs['cell_type'] = data.obs['${params.finetune_label_key}'].tolist()
    data.obs = empty_obs
    data.obs['n_counts'] = data.X.sum(axis=1)

    # ensembl_id 应在数据预处理的步骤加入
    # data.var['ensembl_id'] = data.var['feature_id']

    tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in data.obs.keys()]), nproc=4)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(data)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    tokenized_dataset.save_to_disk('tokenized_dataset')
    """

}

process _finetune_by_langcell {

    tag "${id}"

    label "gpu_task"

    container "housy17/langcell:latest"

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models",
               saveAs: { filename -> "LangCell/${id}" }, enabled: params.finetune_results_dir as boolean
    
    input:
    tuple val(id), path(raw_h5ad), path(tokenized_dataset)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    #!/usr/bin/env python
    import os
    import sys
    sys.path.append(os.path.join("/code", "langcell","data_preprocess"))
    from utils import LangCellTranscriptomeTokenizer
    from datasets import load_from_disk
    from transformers import Trainer
    from transformers import BertForSequenceClassification
    from transformers.training_args import TrainingArguments
    from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
    from collections import Counter

    batch_size = ${params.finetune_batch_size}
    lr_schedule_fn = "linear"
    warmup_steps = 500
    epochs = ${params.finetune_epoch}
    optimizer = "adamw"
    output_dir = "checkpoints"

    dataset = load_from_disk("${tokenized_dataset}")
    trainset_organ_shuffled = dataset.shuffle(seed=1)

    trainset_organ_shuffled = trainset_organ_shuffled.rename_column('cell_type',"label")
    target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))

    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)

    test_size = ${params.finetune_eval_size}
    train_size = round(len(labeled_trainset)*(1-test_size))
    labeled_train_split = labeled_trainset.select([i for i in range(0, train_size)])
    labeled_eval_split = labeled_trainset.select([i for i in range(train_size, len(labeled_trainset))])
    trained_labels = list(Counter(labeled_train_split["label"]).keys())

    def if_trained_label(example):
        return example["label"] in trained_labels
    labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

    train_set = labeled_train_split
    eval_set = labeled_eval_split_subset
    label_name_id = target_name_id_dict

    organ_trainset = train_set
    organ_evalset = eval_set
    organ_label_dict = label_name_id

    id2label = {v: k for k, v in organ_label_dict.items()}
    label2id = organ_label_dict

    logging_steps = round(len(organ_trainset)/batch_size/2)


    MODEL_WEIGHTS_DIR = os.path.join('/data/model_weights', 'LangCell/ckpt')
    CELL_BERT_DIR = os.path.join(MODEL_WEIGHTS_DIR, 'cell_bert')

    model = BertForSequenceClassification.from_pretrained(
        CELL_BERT_DIR,
        num_labels=len(organ_label_dict.keys()),
        id2label=id2label,
        label2id=label2id,
        output_attentions = False,
        output_hidden_states = False
    ).cuda()

    import math
    epoch_step = math.ceil(len(organ_trainset)/batch_size)
    strategy = "steps" if epoch_step > 500 else "epoch"
    
    training_args = {
        "learning_rate": 5e-5,
        "do_train": True,
        "do_eval": True,

        "save_strategy": strategy,   
        "save_steps": 500,          
        "evaluation_strategy": strategy,
        "eval_steps": 500,

        "save_total_limit": 2,
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "output_dir": output_dir,
    }

    training_args_init = TrainingArguments(**training_args)

    from sklearn.metrics import accuracy_score, f1_score
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy and macro f1 using sklearn's function
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        return {
        'accuracy': acc,
        'macro_f1': macro_f1
        }

    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=organ_trainset,
        eval_dataset=organ_evalset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    trainer.save_model(f"./{timestamp}_finetuned_model")
    """

}

process _predict_by_langcell {

    tag "${id}"

    label "gpu_task"

    container "housy17/langcell:latest"

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "LangCell/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(tokenized_dataset), path(finetuned_model)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    #!/usr/bin/env python
    import os
    import sys
    sys.path.append(os.path.join("/code", "langcell","data_preprocess"))
    from datasets import load_from_disk
    from transformers import Trainer
    from transformers import AutoModelForSequenceClassification

    from transformers.training_args import TrainingArguments
    from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
    from scipy.special import softmax

    dataset = load_from_disk("${tokenized_dataset}")
    
    model = AutoModelForSequenceClassification.from_pretrained("${finetuned_model}").cuda()

    args = TrainingArguments(
        output_dir="eval_output",
        per_device_eval_batch_size=${params.predict_batch_size},
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorForCellClassification(),
    )

    pred = trainer.predict(dataset)
    probs = softmax(pred.predictions, axis=-1)
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    import pandas as pd
    df = pd.DataFrame(probs, columns=label_names, index=dataset['barcode'])
    df.to_csv("langcell_predictions.tsv", sep="\\t")
    """

    
}