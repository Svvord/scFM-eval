params.model = "scGPT/scGPT_human"
params.batch_size = 64
params.emb_results_dir = "results"

process embed_by_scgpt {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/scgpt:0.2.4'

    publishDir "${params.emb_results_dir}/embeddings/scgpt", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad) // update: raw_count, data transformed codes inside
    //  log1p-transformed + hgnc symbol

    output:
    tuple val(id), val("scGPT"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import scgpt as scg
    import scanpy as sc
    import os
    import pandas as pd

    adata = sc.read("${raw_h5ad}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    gene_col = "index"
    model_dir = os.path.join("/data/model_weights", "${params.model}")

    adata_embedding = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col = gene_col,
        batch_size = ${params.batch_size},
        obs_to_save = list(adata.obs.columns),
        return_new_adata = True,
    )

    adata_embedding.obs = adata.obs.copy()
    adata_embedding.var = pd.DataFrame(index=[f'V{i+1}' for i in range(adata_embedding.n_vars)])
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write(f"scgpt_embeddings.h5ad")

    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    """
}


params.finetune_batch_size   = 32   // official scGPT annotation batch size
params.finetune_epoch        = 10   // official scGPT annotation epochs
params.finetune_eval_size    = 0.2
params.predict_batch_size    = 64
params.finetune_results_dir  = ""

process finetune_by_scgpt {

    tag "${id}"

    label "gpu_task"

    container 'housy17/scgpt:0.2.4'

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models",
               saveAs: { filename -> "scGPT/${id}" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    #!/usr/bin/env python

    import copy
    import gc
    import json
    import os
    from pathlib import Path
    import shutil
    import sys
    import time
    import traceback
    from typing import List, Tuple, Dict, Union, Optional
    import warnings
    import pandas as pd
    import pickle
    import torch
    from anndata import AnnData
    import scanpy as sc
    import seaborn as sns
    import numpy as np
    from scipy.sparse import issparse
    import matplotlib.pyplot as plt
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from torchtext.vocab import Vocab
    from torchtext._torchtext import (
        Vocab as VocabPybind,
    )
    from sklearn.metrics import confusion_matrix

    sys.path.insert(0, "../")
    import scgpt as scg
    from scgpt.model import TransformerModel, AdversarialDiscriminator
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from scgpt.loss import (
        masked_mse_loss,
        masked_relative_error,
        criterion_neg_log_bernoulli,
    )
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.preprocess import Preprocessor
    from scgpt import SubsetsBatchSampler
    from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

    sc.set_figure_params(figsize=(6, 6))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    hyperparameter_defaults = dict(
        seed=0,
        do_train=True,
        load_model="../save/scGPT_human",
        mask_ratio=0.0,
        epochs=10,
        n_bins=51,
        MVC=False, # Masked value prediction for cell embedding
        ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        layer_size=128,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
        include_zero_gene = False,
        freeze = False, #freeze
        DSBN = False,  # Domain-spec batchnorm
    )

    set_seed(hyperparameter_defaults['seed'])

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = hyperparameter_defaults['mask_ratio']
    mask_value = "auto" 
    include_zero_gene = hyperparameter_defaults["include_zero_gene"]
    max_seq_len = 3001
    n_bins = hyperparameter_defaults['n_bins']

    # input/output representation
    input_style = "binned"  # "normed_raw", "log1p", or "binned"
    output_style = "binned"  # "normed_raw", "log1p", or "binned"

    # settings for training
    MLM = False  # whether to use masked language modeling, currently it is always on.
    CLS = True  # celltype classification objective
    ADV = False  # Adversarial training for batch correction
    CCE = False  # Contrastive cell embedding objective
    MVC = hyperparameter_defaults['MVC']  # Masked value prediction for cell embedding
    ECS = hyperparameter_defaults['ecs_thres'] > 0  # Elastic cell similarity objective
    DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
    input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
    cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 0
    mvc_decoder_style = "inner product"
    ecs_threshold = hyperparameter_defaults['ecs_thres']
    dab_weight = hyperparameter_defaults['dab_weight']

    explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
    do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

    per_seq_batch_sample = False
    # settings for optimizer
    lr = hyperparameter_defaults['lr']  # TODO: test learning rate ratio between two tasks
    lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
    batch_size = ${params.finetune_batch_size}
    eval_batch_size = ${params.finetune_batch_size} * 2
    epochs = ${params.finetune_epoch}
    schedule_interval = 1

    # settings for the model
    fast_transformer = hyperparameter_defaults['fast_transformer']
    fast_transformer_backend = "flash"  # "linear" or "flash"
    embsize = hyperparameter_defaults['layer_size']  # embedding dimension
    d_hid = hyperparameter_defaults['layer_size']  # dimension of the feedforward network in TransformerEncoder
    nlayers = hyperparameter_defaults['nlayers']  # number of TransformerEncoderLayer in TransformerEncoder
    nhead = hyperparameter_defaults['nhead']  # number of heads in nn.MultiheadAttention
    dropout = hyperparameter_defaults['dropout']  # dropout probability

    # logging
    log_interval = 100  # iterations
    save_eval_interval = hyperparameter_defaults['save_eval_interval']  # epochs
    do_eval_scib_metrics = True



    # %% validate settings
    assert input_style in ["normed_raw", "log1p", "binned"]
    assert output_style in ["normed_raw", "log1p", "binned"]
    assert input_emb_style in ["category", "continuous", "scaling"]
    if input_style == "binned":
        if input_emb_style == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
    elif input_style == "log1p" or input_style == "normed_raw":
        if input_emb_style == "category":
            raise ValueError(
                "input_emb_style `category` is not supported for log1p or normed_raw input."
            )

    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins  # for padding gene expr values
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins

    if ADV and DAB:
        raise ValueError("ADV and DAB cannot be both True.")
    DAB_separate_optim = True if DAB > 1 else False


    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    save_dir = Path(f"./{timestamp}_finetuned_model")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger


    # ============= load model ============== #

    model_dir = Path("/data/model_weights/${params.model}")
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    shutil.copy(model_config_file, save_dir / "args.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    # ============= load data =============== #

    adata = sc.read_h5ad("${raw_h5ad}")
    celltype_id_labels = adata.obs["${params.finetune_label_key}"].astype("category").cat.codes.values
    celltypes = adata.obs["${params.finetune_label_key}"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["${params.finetune_label_key}"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()


    # ============ process data ============= #
    
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3", # if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=None)

    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()
    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    # Crash-safe stratified split: every class keeps >=1 training sample; a class too
    # small to spare one for validation goes entirely to train (benchmark fairness).
    def _min_train_split(_labels, _vf, _seed=42):
        _labels = np.asarray(_labels); _rng = np.random.RandomState(_seed)
        _tr, _va = [], []
        for _c in np.unique(_labels):
            _ix = np.where(_labels == _c)[0]; _rng.shuffle(_ix)
            _nv = min(int(len(_ix) * _vf), len(_ix) - 1)
            _va.extend(_ix[:_nv].tolist()); _tr.extend(_ix[_nv:].tolist())
        _rng.shuffle(_tr); _rng.shuffle(_va)
        return np.array(_tr, dtype=int), np.array(_va, dtype=int)
    _tr_idx, _va_idx = _min_train_split(celltypes_labels, ${params.finetune_eval_size}, 42)
    train_data, valid_data = all_counts[_tr_idx], all_counts[_va_idx]
    train_celltype_labels, valid_celltype_labels = celltypes_labels[_tr_idx], celltypes_labels[_va_idx]

    # ============================= #

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\\n\\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\\n\\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )
        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )

        tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
        tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "celltype_labels": tensor_celltype_labels_train,
        }
        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "celltype_labels": tensor_celltype_labels_valid,
        }

        return train_data_pt, valid_data_pt


    # dataset
    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}


    # data_loader
    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        if num_workers == 0:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

        dataset = SeqDataset(data_pt)

        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    # ========== load model ============= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=0,
        domain_spec_batchnorm=hyperparameter_defaults['DSBN'],
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=hyperparameter_defaults['pre_norm'],
    )

    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        print("-"*20)
        print(f"name: {name}")
        if hyperparameter_defaults['freeze'] and "encoder" in name and "transformer_encoder" not in name:
        # if config.freeze and "encoder" in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")

    model.to(device)
    if ADV:
        discriminator = AdversarialDiscriminator(
            d_model=embsize,
            n_cls=num_batch_types,
        ).to(device)


    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-4 if hyperparameter_defaults['amp'] else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, schedule_interval, gamma=hyperparameter_defaults['schedule_ratio']
    )
    if DAB_separate_optim:
        optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler_dab = torch.optim.lr_scheduler.StepLR(
            optimizer_dab, schedule_interval, gamma=hyperparameter_defaults['schedule_ratio']
        )
    if ADV:
        criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
        optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, schedule_interval, gamma=hyperparameter_defaults['schedule_ratio']
        )
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, schedule_interval, gamma=hyperparameter_defaults['schedule_ratio']
        )

    scaler = torch.cuda.amp.GradScaler(enabled=hyperparameter_defaults['amp'])


    def train(model: nn.Module, loader: DataLoader) -> None:
        model.train()
        (
            total_loss,
            total_mse,
            total_cls,
            total_cce,
            total_mvc,
            total_ecs,
            total_dab,
            total_adv_E,
            total_adv_D,
            total_zero_log_prob,
            total_mvc_zero_log_prob,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        total_error = 0.0
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=hyperparameter_defaults['amp']):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=do_sample_in_train,
                    #generative_training=False
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict
                loss = 0.0
                metrics_to_log = {}
                if MLM:
                    loss_mse = criterion(
                        output_dict["mlm_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mse
                    metrics_to_log = {"train/mse": loss_mse.item()}
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if CLS:
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss = loss + loss_cls
                    metrics_to_log.update({"train/cls": loss_cls.item()})

                    error_rate = 1 - (
                        (output_dict["cls_output"].argmax(1) == celltype_labels)
                        .sum()
                        .item()
                    ) / celltype_labels.size(0)
                if CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    metrics_to_log.update({"train/cce": loss_cce.item()})
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc
                    metrics_to_log.update({"train/mvc": loss_mvc.item()})
                if MVC and explicit_zero_prob:
                    loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc_zero_log_prob
                    metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
                if ECS:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                if DAB:
                    # try weighting and separate optimizer
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                    loss = loss + dab_weight * loss_dab
                    metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_mse += loss_mse.item() if MLM else 0.0
            total_cls += loss_cls.item() if CLS else 0.0
            total_cce += loss_cce.item() if CCE else 0.0
            total_mvc += loss_mvc.item() if MVC else 0.0
            total_ecs += loss_ecs.item() if ECS else 0.0
            total_dab += loss_dab.item() if DAB else 0.0
            total_adv_E += loss_adv_E.item() if ADV else 0.0
            total_adv_D += loss_adv_D.item() if ADV else 0.0
            total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
            total_mvc_zero_log_prob += (
                loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
            )
            total_error += error_rate
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_cls = total_cls / log_interval if CLS else 0.0
                cur_cce = total_cce / log_interval if CCE else 0.0
                cur_mvc = total_mvc / log_interval if MVC else 0.0
                cur_ecs = total_ecs / log_interval if ECS else 0.0
                cur_dab = total_dab / log_interval if DAB else 0.0
                cur_adv_E = total_adv_E / log_interval if ADV else 0.0
                cur_adv_D = total_adv_D / log_interval if ADV else 0.0
                cur_zero_log_prob = (
                    total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
                )
                cur_mvc_zero_log_prob = (
                    total_mvc_zero_log_prob / log_interval
                    if MVC and explicit_zero_prob
                    else 0.0
                )
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                    + (f"cls {cur_cls:5.2f} | " if CLS else "")
                    + (f"err {cur_error:5.2f} | " if CLS else "")
                    + (f"cce {cur_cce:5.2f} |" if CCE else "")
                    + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                    + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                    + (f"dab {cur_dab:5.2f} |" if DAB else "")
                    + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                    + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                    + (
                        f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                        if MVC and explicit_zero_prob
                        else ""
                    )
                )
                total_loss = 0
                total_mse = 0
                total_cls = 0
                total_cce = 0
                total_mvc = 0
                total_ecs = 0
                total_dab = 0
                total_adv_E = 0
                total_adv_D = 0
                total_zero_log_prob = 0
                total_mvc_zero_log_prob = 0
                total_error = 0
                start_time = time.time()
    def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        predictions = []
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=hyperparameter_defaults['amp']):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=None,
                        CLS=CLS,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=do_sample_in_train,
                        #generative_training = False,
                    )
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)

                    if DAB:
                        loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)

        if return_raw:
            return np.concatenate(predictions, axis=0)

        return total_loss / total_num, total_error / total_num

    
    # ========= fine-tune ============ #
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    # define_wandb_metrcis()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=eval_batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )


        train(
            model,
            loader=train_loader,
        )

        val_loss, val_err = evaluate(
            model,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        scheduler.step()
        if DAB_separate_optim:
            scheduler_dab.step()
    
    label2id = {v:k for k, v in id2type.items()}

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    with open(save_dir / "label_map.json", "w") as f:
        json.dump(
            {
                "id2label": id2type,
                "label2id": label2id,
            },
            f
        )

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    """
}



process predict_by_scgpt {

    tag "${id}"

    label "gpu_task"

    container 'housy17/scgpt:0.2.4'

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "scGPT/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(test_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    #!/usr/bin/env python
    import os
    import sys
    import copy
    import gc
    import json
    from pathlib import Path
    import scanpy as sc
    import numpy as np
    from scipy.sparse import issparse
    from typing import List, Tuple, Dict, Union, Optional

    import scgpt as scg
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.preprocess import Preprocessor
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from scgpt.model import TransformerModel, AdversarialDiscriminator

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader

    # =========== hyperparameters =========== #
    eval_batch_size = ${params.predict_batch_size}
    n_bins = 51
    max_seq_len = 3001
    mask_ratio=0.0
    mask_value = "auto"
    amp=True
    pre_norm=False
    DSBN = False
    MLM = False  # whether to use masked language modeling, currently it is always on.
    CLS = True  # celltype classification objective
    ADV = False  # Adversarial training for batch correction
    CCE = False  # Contrastive cell embedding objective
    MVC = False
    DAB = False
    INPUT_BATCH_LABELS = False
    include_zero_gene = False
    cell_emb_style = "cls"
    input_emb_style = "continuous" 
    input_layer_key = "X_binned"
    mvc_decoder_style = "inner product"
    ecs_threshold = 0.0
    ECS = ecs_threshold > 0
    dab_weight = 0.0
    dropout = 0.2 # useless, because model.eval()
    explicit_zero_prob = MLM and include_zero_gene
    do_sample_in_train = False and explicit_zero_prob
    per_seq_batch_sample = False
    fast_transformer = True
    fast_transformer_backend = "flash"
    
    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins  # for padding gene expr values
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins

    logger = scg.logger

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    model_dir = Path("${model_weights}")
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    model_file = model_dir / "best_model.pt"
    model_config_file = model_dir / "args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    # ============= load data =============== #
    adata = sc.read_h5ad("${test_h5ad}")
    adata.var["gene_name"] = adata.var.index.tolist()

    # ============ process data ============= #
    
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3", # if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=None)

    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    test_data = all_counts

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_test = tokenize_and_pad_batch(
        test_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )

    # dataset
    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    # ============= load model ============== #

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map_file = model_dir / "label_map.json"
    with open(label_map_file, "r") as f:
        tmp = json.load(f)
        id2label = tmp["id2label"]
        label2id = tmp["label2id"]
    id2label = {int(k): v for k, v in id2label.items()}
    num_types = len(id2label)


    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=0,
        domain_spec_batchnorm=DSBN,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=pre_norm,
    )

    model.load_state_dict(torch.load(model_file))
    logger.info(f"Loading all model params from {model_file}")
    model.to(device)

    pred_proba = np.zeros((len(adata), num_types))
    i = 0
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                )
                output_values = output_dict["cls_output"]

            pred_proba[i*eval_batch_size:(i+1)*eval_batch_size] = output_values.cpu().numpy()
            i+= 1

    from scipy.special import softmax
    probs = softmax(pred_proba, axis=-1)
    label_names = [id2label[i] for i in range(len(id2label))]
    import pandas as pd
    df = pd.DataFrame(probs, columns=label_names, index=adata.obs['barcode'].tolist())
    df.to_csv("scgpt_predictions.tsv", sep="\\t")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    """


}


// ============================ batch integration ============================ //
// Native scGPT batch integration (upstream Tutorial_Integration recipe): fine-tune
// the pretrained model with DSBN (domain-specific batchnorm keyed on batch) + DAB
// (domain adaptation by reverse backprop / adversarial-on-batch) + GEPC/MVC + ECS,
// using ONLY batch labels. The cell-type classifier (CLS) is OFF, so no cell type
// enters training/inference. The original obs (incl. cell_type + the string
// batch_id) is preserved for the downstream scIB benchmark. The integrated cell
// embedding is extracted via model.encode_batch with cells SORTED by batch (DSBN
// applies one batch's stats per chunk using batch_labels[0]).

params.batch_key              = "batch_id"
params.integration_epoch      = 15
params.integration_batch_size = 64
params.integration_n_hvg      = 1200

process integrate_by_scgpt {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/scgpt:0.2.4'

    publishDir "${params.emb_results_dir}/embeddings/scgpt_integrated", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("scGPT (integrated)"), path("*embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import copy
    import json
    import os
    import sys
    import warnings
    from pathlib import Path
    from typing import Dict, Tuple

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    from scipy.sparse import issparse
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import Dataset, DataLoader

    import scgpt as scg
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.preprocess import Preprocessor
    from scgpt import SubsetsBatchSampler
    from scgpt.utils import set_seed

    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings("ignore")
    set_seed(0)
    logger = scg.logger

    # ----------------------- integration config ----------------------- #
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = 0.4
    mask_value = -1
    pad_value = -2
    n_bins = 51
    n_input_bins = n_bins
    n_hvg = ${params.integration_n_hvg}
    max_seq_len = n_hvg + 1

    input_emb_style = "continuous"
    cell_emb_style = "cls"
    mvc_decoder_style = "inner product"

    MLM = True               # masked expression reconstruction (main objective)
    CLS = False              # cell-type classifier OFF -> no label leakage
    CCE = False
    MVC = True               # GEPC: masked value prediction for cell embedding
    ECS = True               # elastic cell similarity
    DAB = True               # domain adaptation by reverse backprop (on batch)
    INPUT_BATCH_LABELS = True
    DSBN = True              # domain-specific batchnorm (on batch)
    per_seq_batch_sample = True
    explicit_zero_prob = True
    do_sample_in_train = False
    include_zero_gene = True  # official integration recipe: include all HVGs, including zeros

    ecs_threshold = 0.8
    dab_weight = 1.0

    lr = 1e-4
    batch_size = ${params.integration_batch_size}
    eval_batch_size = ${params.integration_batch_size} * 2
    epochs = ${params.integration_epoch}
    schedule_interval = 1
    schedule_ratio = 0.9
    amp = True
    log_interval = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------- load model --------------------------- #
    model_dir = Path("/data/model_weights/${params.model}")
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    # ---------------------------- load data --------------------------- #
    adata = sc.read_h5ad("${raw_h5ad}")
    original_obs = adata.obs.copy()
    original_spatial = adata.obsm["spatial"].copy() if "spatial" in adata.obsm else None

    if "${params.batch_key}" not in adata.obs:
        raise KeyError("scGPT integration requires adata.obs['${params.batch_key}']")

    adata.var["gene_name"] = adata.var.index.tolist()
    adata.obs["str_batch"] = adata.obs["${params.batch_key}"].astype(str)
    adata.obs["batch_id_code"] = adata.obs["str_batch"].astype("category").cat.codes.values
    num_batch_types = int(pd.unique(adata.obs["batch_id_code"]).shape[0])

    # Gracefully degrade to a label-free fine-tune if there is nothing to integrate.
    if num_batch_types < 2:
        logger.info("Fewer than 2 batches; disabling batch-correction objectives.")
        DAB = False
        DSBN = False
        INPUT_BATCH_LABELS = False
        per_seq_batch_sample = False

    # keep only genes present in the model vocabulary (subsets var, never obs)
    adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # seurat_v3 HVG needs scikit-misc; fall back to cell_ranger if unavailable.
    try:
        import skmisc  # noqa: F401
        hvg_flavor = "seurat_v3"
    except Exception:
        hvg_flavor = "cell_ranger"
        logger.info("scikit-misc not found; using cell_ranger HVG flavor.")

    def _build_preprocessor(flavor):
        return Preprocessor(
            use_key="X",
            filter_gene_by_counts=3,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            subset_hvg=n_hvg,
            hvg_flavor=flavor,
            binning=n_bins,
            result_binned_key="X_binned",
        )

    # seurat_v3 batch-aware HVG occasionally hits a LOESS singularity on some
    # datasets ("reciprocal condition number ~0"); fall back to the robust
    # cell_ranger flavor (no batch_key) so one tissue can't abort the run.
    adata_pre = adata.copy()
    try:
        _build_preprocessor(hvg_flavor)(adata, batch_key="str_batch")
    except Exception as exc:
        logger.info(f"HVG flavor={hvg_flavor} (batch-aware) failed ({exc}); retrying cell_ranger, no batch_key.")
        adata = adata_pre.copy()
        _build_preprocessor("cell_ranger")(adata, batch_key=None)
    del adata_pre

    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()
    batch_labels_all = adata.obs["batch_id_code"].values.astype(int)

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    (
        train_data,
        valid_data,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, batch_labels_all, test_size=${params.finetune_eval_size}, shuffle=True
    )

    tokenized_train = tokenize_and_pad_batch(
        train_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value, append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value, append_cls=True,
        include_zero_gene=include_zero_gene,
    )

    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"], mask_ratio=mask_ratio,
            mask_value=mask_value, pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"], mask_ratio=mask_ratio,
            mask_value=mask_value, pad_value=pad_value,
        )
        train_data_pt = {
            "gene_ids": tokenized_train["genes"],
            "values": masked_values_train,
            "target_values": tokenized_train["values"],
            "batch_labels": torch.from_numpy(train_batch_labels).long(),
        }
        valid_data_pt = {
            "gene_ids": tokenized_valid["genes"],
            "values": masked_values_valid,
            "target_values": tokenized_valid["values"],
            "batch_labels": torch.from_numpy(valid_batch_labels).long(),
        }
        if sort_seq_batch:
            train_sort_ids = np.argsort(train_batch_labels)
            train_data_pt = {k: v[train_sort_ids] for k, v in train_data_pt.items()}
            valid_sort_ids = np.argsort(valid_batch_labels)
            valid_data_pt = {k: v[valid_sort_ids] for k, v in valid_data_pt.items()}
        return train_data_pt, valid_data_pt

    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    def prepare_dataloader(data_pt, batch_size, shuffle=False,
                           intra_domain_shuffle=False, drop_last=False, num_workers=0):
        if num_workers == 0:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
        dataset = SeqDataset(data_pt)
        if per_seq_batch_sample:
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            return DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets, batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle, drop_last=drop_last,
                ),
                num_workers=num_workers, pin_memory=True,
            )
        return DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            drop_last=drop_last, num_workers=num_workers, pin_memory=True,
        )

    # --------------------------- build model -------------------------- #
    ntokens = len(vocab)
    model = TransformerModel(
        ntokens, embsize, nhead, d_hid, nlayers,
        nlayers_cls=3, n_cls=1, vocab=vocab, dropout=0.2,
        pad_token=pad_token, pad_value=pad_value,
        do_mvc=MVC, do_dab=DAB, use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types, domain_spec_batchnorm=DSBN,
        input_emb_style=input_emb_style, n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style, mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold, explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=True, fast_transformer_backend="flash", pre_norm=False,
    )

    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loaded all model params from {model_file}")
    except Exception:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained params.")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(device)

    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if amp else 1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def train(model, loader):
        model.train()
        running = 0.0
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if (DSBN or INPUT_BATCH_LABELS) else None,
                    CLS=False, CCE=CCE, MVC=MVC, ECS=ECS, do_sample=do_sample_in_train,
                )
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
                if explicit_zero_prob:
                    loss = loss + criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                if MVC:
                    loss = loss + criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                if MVC and explicit_zero_prob:
                    loss = loss + criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                if ECS:
                    loss = loss + 10.0 * output_dict["loss_ecs"]
                if DAB:
                    loss = loss + dab_weight * criterion_dab(
                        output_dict["dab_output"], batch_labels
                    )
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if batch % log_interval == 0 and batch > 0:
                logger.info(f"epoch {epoch:3d} | {batch:4d}/{len(loader):4d} | loss {running / log_interval:6.3f}")
                running = 0.0

    def evaluate(model, loader):
        model.eval()
        total_loss = 0.0
        total_num = 0
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=amp):
                    output_dict = model(
                        input_gene_ids, input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if (DSBN or INPUT_BATCH_LABELS) else None,
                        CLS=False, CCE=False, MVC=False, ECS=False, do_sample=do_sample_in_train,
                    )
                    masked_positions = input_values.eq(mask_value)
                    loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
                total_loss += loss.item() * len(input_gene_ids)
                total_num += len(input_gene_ids)
        return total_loss / max(total_num, 1)

    # ----------------------------- fine-tune -------------------------- #
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(1, epochs + 1):
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt, batch_size=batch_size, shuffle=False,
            intra_domain_shuffle=True, drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt, batch_size=eval_batch_size, shuffle=False,
            intra_domain_shuffle=False, drop_last=False,
        )
        train(model, train_loader)
        val_loss = evaluate(model, valid_loader)
        logger.info(f"end of epoch {epoch:3d} | valid mse {val_loss:6.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        scheduler.step()

    if best_model is not None:
        model = best_model

    # ----------------- extract integrated embedding ------------------- #
    # Cells MUST be grouped by batch: encode_batch applies DSBN per chunk using
    # batch_labels[0], so each batch_size chunk must be single-batch.
    model.eval()
    if per_seq_batch_sample:
        sort_idx = np.argsort(adata.obs["batch_id_code"].values)
    else:
        sort_idx = np.arange(adata.n_obs)
    adata_sorted = adata[sort_idx]
    all_counts_emb = (
        adata_sorted.layers[input_layer_key].toarray()
        if issparse(adata_sorted.layers[input_layer_key])
        else adata_sorted.layers[input_layer_key]
    )
    batch_ids_emb = adata_sorted.obs["batch_id_code"].values.astype(int)

    tokenized_all = tokenize_and_pad_batch(
        all_counts_emb, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value, append_cls=True, include_zero_gene=True,
    )
    all_gene_ids = tokenized_all["genes"]
    all_values = tokenized_all["values"]
    src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        cell_embeddings = model.encode_batch(
            all_gene_ids, all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=eval_batch_size,
            batch_labels=torch.from_numpy(batch_ids_emb).long() if (DSBN or INPUT_BATCH_LABELS) else None,
            time_step=0, return_np=True,
        )
    cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)

    import anndata as ad
    obs_out = original_obs.iloc[sort_idx].copy()
    adata_embedding = ad.AnnData(
        X=np.asarray(cell_embeddings, dtype=np.float32),
        obs=obs_out,
        var=pd.DataFrame(index=[f"V{i+1}" for i in range(cell_embeddings.shape[1])]),
    )
    if original_spatial is not None:
        adata_embedding.obsm["spatial"] = original_spatial[sort_idx]
    adata_embedding.write("scgpt_integrated_embeddings.h5ad")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    """
}
