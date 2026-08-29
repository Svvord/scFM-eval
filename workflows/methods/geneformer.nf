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
    import numpy as np
    from pathlib import Path
    from scipy.sparse import issparse, csr_matrix

    adata = sc.read_h5ad("${raw_h5ad}")
    adata.obs['n_counts'] = np.asarray(adata.X.sum(axis=1)).ravel()
    if 'barcode' not in adata.obs:
        adata.obs['barcode'] = adata.obs_names.tolist()
    if 'filter_pass' in adata.obs:
        adata.obs.drop(columns=['filter_pass'], inplace=True)
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

    model_name = os.path.basename("${params.model}".rstrip("/"))
    if "V1" in model_name:
        model_version = "V1"
    elif "V2" in model_name:
        model_version = "V2"
    else:
        raise ValueError(f"Cannot infer Geneformer model version from model path: {model_name}")
    emb_mode = "cls" if model_version == "V2" else "cell"

    tk = TranscriptomeTokenizer(
        {'barcode': 'barcode'},
        nproc=16,
        model_version=model_version,
    )
    tk.tokenize_data(
        './preproc',            # "data_directory"
        './',                   # "output_directory"
        "transformed",          # "output_prefix"
        file_format="h5ad"
    )
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_mode=emb_mode,
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
    if 'barcode' in obs.columns:
        expected_index = obs['barcode'].astype(str).tolist()
    else:
        expected_index = obs.index.astype(str).tolist()
    missing = [item for item in expected_index if item not in embedding.index]
    if missing:
        raise KeyError(f"{len(missing)} cells are missing from Geneformer embeddings; first missing barcode: {missing[0]}")
    embedding = embedding.loc[expected_index, :].values

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


params.finetune_batch_size   = 4
params.finetune_epoch        = 1
params.finetune_eval_size    = 0.1
params.predict_batch_size    = 16
params.finetune_results_dir  = ""

process _finetune_by_geneformer {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/geneformer:latest'

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models", mode: 'copy',
               saveAs: { filename -> "geneformer/${id}" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(processed_raw_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    #!/usr/bin/env python
    import os
    from geneformer import Classifier
    from geneformer import TranscriptomeTokenizer
    from pathlib import Path

    # Geneformer splits train/valid internally via HF Dataset.train_test_split(
    # stratify_by_column=...), which raises on singleton classes. Patch it (local to
    # this process) to a crash-safe stratified split consistent with the other methods:
    # every class keeps >=1 training sample; a class too small to spare one for
    # validation goes entirely to train. Non-stratified calls fall through unchanged.
    import numpy as _np
    import datasets as _hfds
    _orig_tts = _hfds.Dataset.train_test_split
    def _safe_train_test_split(self, test_size=None, stratify_by_column=None, seed=42, **_kw):
        if stratify_by_column is None or test_size is None:
            return _orig_tts(self, test_size=test_size,
                             stratify_by_column=stratify_by_column, seed=seed, **_kw)
        _labels = _np.asarray(self[stratify_by_column]); _rng = _np.random.RandomState(42 if seed is None else seed)
        _tr, _va = [], []
        for _c in _np.unique(_labels):
            _ix = _np.where(_labels == _c)[0]; _rng.shuffle(_ix)
            _nv = min(int(len(_ix) * test_size), len(_ix) - 1)
            _va.extend(_ix[:_nv].tolist()); _tr.extend(_ix[_nv:].tolist())
        _rng.shuffle(_tr); _rng.shuffle(_va)
        return _hfds.DatasetDict({"train": self.select(_tr), "test": self.select(_va)})
    _hfds.Dataset.train_test_split = _safe_train_test_split

    model_name = os.path.basename("${params.model}".rstrip("/"))
    if "V1" in model_name:
        model_version = "V1"
    elif "V2" in model_name:
        model_version = "V2"
    else:
        raise ValueError(f"Cannot infer Geneformer model version from model path: {model_name}")

    tk = TranscriptomeTokenizer(
        {'barcode': 'barcode', '${params.finetune_label_key}': 'cell_type'}, 
        nproc=16,
        model_version=model_version,
    )

    tk.tokenize_data(
        './preproc',            # "data_directory"
        './',                   # "output_directory"
        "transformed",          # "output_prefix"
        file_format="h5ad"
    )

    # Geneformer trains by STEPS. Per-tissue references are small, so 1 epoch can be
    # only a few hundred steps -- too few updates, and an absolute warmup would never
    # complete. Floor training at GENEFORMER_MIN_STEPS gradient steps; larger tissues
    # keep their ~1-epoch budget. warmup_ratio scales warmup to the actual step count.
    import math as _math
    GENEFORMER_MIN_STEPS = 1000
    _n_total = len(_hfds.load_from_disk("transformed.dataset"))
    _n_train = int(round(_n_total * (1 - ${params.finetune_eval_size})))
    _steps_per_epoch = max(1, _math.ceil(_n_train / ${params.finetune_batch_size}))
    _max_steps = max(GENEFORMER_MIN_STEPS, _steps_per_epoch * ${params.finetune_epoch})
    _eval_steps = max(50, _max_steps // 10)   # ~10 eval/save points for best-model selection
    print(f"[geneformer] n_total={_n_total} n_train={_n_train} steps/epoch={_steps_per_epoch} -> max_steps={_max_steps} eval_steps={_eval_steps}")

    cell_state_dict = {
        "state_key": "cell_type",
        "states": "all",
    }

    training_args = {
        "max_steps": _max_steps,                 # step floor (>=1000) instead of a raw epoch count
        "per_device_train_batch_size": ${params.finetune_batch_size},
        "lr_scheduler_type": "cosine",
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,                     # warmup = 10% of actual steps (scales per tissue)
        # Step-based eval + return best checkpoint (NO early stopping). Epoch-based eval
        # gave large tissues only one eval point; step-based gives ~10 across any tissue
        # size, so load_best_model_at_end actually selects a best checkpoint.
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": _eval_steps,
        "save_steps": _eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "seed": 1
    }

    valid_size = ${params.finetune_eval_size}
    train_size = 1 - valid_size
    cc = Classifier(
        classifier="cell",                 # 做 cell-state / cell-type 分类
        cell_state_dict=cell_state_dict,
        training_args=training_args,
        max_ncells=None,
        freeze_layers=2,                   # 冻结前几层，减轻过拟合，可按需改
        num_crossval_splits=1,             # 简单 train/val
        split_sizes={"train": train_size, "valid": valid_size, "test": 0.0},
        forward_batch_size= 2 * ${params.finetune_batch_size},            # eval 时 batch size
        model_version=model_version,
        nproc=16,
        ngpu=1,
    )

    Path('./train/').mkdir()
    cc.prepare_data(
        input_data_file="transformed.dataset",
        output_directory="./train",
        output_prefix="annotation_task",
    )

    cc.validate(
        model_directory = "/data/model_weights/${params.model}",
        prepared_input_data_file = "./train/annotation_task_labeled.dataset",
        id_class_dict_file = "./train/annotation_task_id_class_dict.pkl",
        output_directory = "./train",
        output_prefix = "annotation_task"
    )

    import shutil
    import glob
    Path('./geneformer_finetuned_model').mkdir()

    files = glob.glob('train/*_geneformer_cellClassifier_annotation_task/ksplit1/config.json')
    shutil.move(files[0], "./geneformer_finetuned_model/config.json")
    files = glob.glob('train/*_geneformer_cellClassifier_annotation_task/ksplit1/model.safetensors')
    shutil.move(files[0], "./geneformer_finetuned_model/model.safetensors")
    files = glob.glob('train/*_geneformer_cellClassifier_annotation_task/ksplit1/training_args.bin')
    shutil.move(files[0], "./geneformer_finetuned_model/training_args.bin")
    shutil.move("./train/annotation_task_id_class_dict.pkl", "./geneformer_finetuned_model/annotation_task_id_class_dict.pkl")
    """
}


process _predict_by_geneformer {

    tag "${id}"

    label "gpu_task"

    container 'housy17/geneformer:latest'

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "geneformer/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(test_h5ad), path(processed_raw_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    #!/usr/bin/env python
    import os
    from geneformer import Classifier
    from geneformer import TranscriptomeTokenizer

    model_name = os.path.basename("${params.model}".rstrip("/"))
    if "V1" in model_name:
        model_version = "V1"
    elif "V2" in model_name:
        model_version = "V2"
    else:
        raise ValueError(f"Cannot infer Geneformer model version from model path: {model_name}")

    tk = TranscriptomeTokenizer(
        {'barcode': 'barcode'}, 
        nproc=16,
        model_version=model_version,
    )

    tk.tokenize_data(
        './preproc',            # "data_directory"
        './',                   # "output_directory"
        "transformed",          # "output_prefix"
        file_format="h5ad"
    )

    cc = Classifier(
        classifier="cell",
        cell_state_dict = {"state_key": "${params.finetune_label_key}", "states": "all"},
        forward_batch_size=${params.predict_batch_size},
        model_version=model_version,
        nproc=16
    )

    import pickle
    from geneformer import classifier_utils as cu
    from geneformer import perturber_utils as pu
    from geneformer import evaluation_utils as eu
    from tqdm.auto import trange
    import torch
    import datasets
    import numpy as np
    

    model_directory="${model_weights}"
    id_class_dict_file="${model_weights}/annotation_task_id_class_dict.pkl"
    test_data_file="transformed.dataset"
    forward_batch_size = cc.forward_batch_size

    with open(id_class_dict_file, "rb") as f:
        id_class_dict = pickle.load(f)
    num_classes = cu.get_num_classes(id_class_dict)

    test_data = pu.load_and_filter(None, cc.nproc, test_data_file)

    model = pu.load_model(
        cc.model_type,
        num_classes,
        model_directory,
        "eval",
        quantize=cc.quantize,
    )

    labels = id_class_dict.keys()

    predict_logits = []
    model.eval()
    evalset_len = len(test_data)
    label_name = 'label'
    max_evalset_len = max(test_data.select([i for i in range(evalset_len)])["length"])

    def preprocess_classifier_batch(cell_batch, max_len, label_name, gene_token_dict):
        if max_len is None:
            max_len = max([len(i) for i in cell_batch["input_ids"]])

        def pad_label_example(example):
            example["input_ids"] = np.pad(
                example["input_ids"],
                (0, max_len - len(example["input_ids"])),
                mode="constant",
                constant_values=gene_token_dict.get("<pad>"),
            )
            example["attention_mask"] = (
                example["input_ids"] != gene_token_dict.get("<pad>")
            ).astype(int)
            return example

        padded_batch = cell_batch.map(pad_label_example)
        return padded_batch

    for i in trange(0, evalset_len, forward_batch_size):
        max_range = min(i + forward_batch_size, evalset_len)
        batch_evalset = test_data.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(
            batch_evalset, max_evalset_len, label_name, cc.gene_token_dict
        )
        padded_batch.set_format(type="torch")

        # For datasets>=4.0.0, convert to dict to avoid format issues
        if int(datasets.__version__.split(".")[0]) >= 4:
            padded_batch = padded_batch[:]
        
        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        # label_batch = padded_batch[label_name]
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_batch.to("cuda"),
                attention_mask=attn_msk_batch.to("cuda"),
                labels=None,
            )
            predict_logits += [outputs.logits.detach().to("cpu")]
    
    logits_by_cell = torch.cat(predict_logits, dim=0)
    last_dim = len(logits_by_cell.shape) - 1
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[last_dim])

    id2label = id_class_dict
    barcodes = list(test_data['barcode'])
    from scipy.special import softmax
    probs = softmax(all_logits.numpy(), axis=-1)
    label_names = [id2label[i] for i in range(len(id2label))]
    import pandas as pd
    df = pd.DataFrame(probs, columns=label_names, index=barcodes)
    df.to_csv("geneformer_predictions.tsv", sep="\\t")
    """

}
