// Hugging Face jkobject/scPRINT `medium-v1.5.ckpt` pinned to the rename commit (see
// workflows/utils/download.nf); it is byte-identical to the checkpoint originally released
// as `v2-medium.ckpt`.
params.model = "scPRINT/medium-v1.5.ckpt"
params.batch_size = 64
params.emb_results_dir = "results"
params.how = "random expr"
params.max_len = 4000
params.scprint_save_every = 10000

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
    
    mkdir -p "\$HOME/.lamin"
    cp -a "\${SCPRINT_LAMIN_SEED:-/opt/scprint/lamin}/.lamin/." "\$HOME/.lamin/"
    
    python - << 'EOF'
    import torch
    import os
    import sys
    sys.path.append(os.path.join("/code", "scprint"))
    from scprint import scPrint
    from scdataloader import Preprocessor, utils
    from scdataloader.utils import load_genes
    from scprint.tasks import GNInfer, Embedder, Denoiser, withknn
    from scprint.model import utils as scprint_model_utils
    
    model_checkpoint_file = os.path.join("/data/model_weights", "${params.model}")

    # make sure that you check if you have a GPU with flashattention or not (see README)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = torch.load(model_checkpoint_file, map_location=device)
    

    transformer = "normal" if device.type == "cpu" else "flash"

    # both are for compatibility issues with different versions of the pretrained model, so we need to load it with the correct transformer
    if "prenorm" in m["hyper_parameters"]:
        # `prenorm` is a legacy hyper-parameter that scPrint.__init__ no longer accepts
        # (pre-norm is the fixed default). Drop it from a TASK-LOCAL copy of the
        # checkpoint; never rewrite the shared file under /data/model_weights.
        m["hyper_parameters"].pop("prenorm")
        model_checkpoint_file = os.path.abspath("scprint_checkpoint_local.ckpt")
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
    if device.type == "cpu":
        model = model.to(torch.float32)

    # you can perform your inference on float16 if you have a GPU, otherwise use float64
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    # the models are often loaded with some parts still displayed as "cuda" and some as "cpu", so we need to make sure that the model is fully on the right device
    model = model.to(device)

    def make_embedding_only_adata(
        pos,
        expr_pred,
        genes,
        embs,
        classes,
        pred=None,
        attention=None,
        label_decoders=None,
        labels_hierarchy={},
        gtclass=None,
        doplot=True,
    ):
        # Keep only outputs needed by this workflow; skip dense expression layers.
        import numpy as np
        import pandas as pd
        from anndata import AnnData

        if embs is None:
            return AnnData(X=np.empty((0, 0), dtype=np.float32)), None

        emb_array = embs.detach().to(device="cpu", dtype=torch.float32).numpy()
        obs = pd.DataFrame(index=range(emb_array.shape[0]))

        if pred is not None and len(classes) > 0:
            pred_array = pred.detach().to(device="cpu", dtype=torch.int32).numpy()
            if pred_array.ndim == 1:
                pred_array = pred_array.reshape(-1, 1)
            columns = ["pred_" + cls for cls in classes]
            if label_decoders is not None:
                pred_array = np.array(
                    [
                        [
                            label_decoders[classes[col_idx]][int(value)]
                            for value in pred_array[:, col_idx]
                        ]
                        for col_idx in range(pred_array.shape[1])
                    ]
                ).T
            obs = pd.DataFrame(pred_array, columns=columns)

        adata = AnnData(X=emb_array, obs=obs)
        adata.obsm["scprint_emb"] = emb_array
        return adata, None

    scprint_model_utils.make_adata = make_embedding_only_adata

    import anndata as ad
    import importlib

    original_anndata_concat = ad.concat

    def concat_without_empty_scprint_shards(adatas, *args, **kwargs):
        if isinstance(adatas, dict):
            filtered = {
                key: value
                for key, value in adatas.items()
                if not (hasattr(value, "n_obs") and value.n_obs == 0)
            }
            if filtered and len(filtered) != len(adatas):
                print(f"Skipping {len(adatas) - len(filtered)} empty scPRINT prediction shard(s).")
                adatas = filtered
        elif isinstance(adatas, (list, tuple)):
            filtered = [
                value
                for value in adatas
                if not (hasattr(value, "n_obs") and value.n_obs == 0)
            ]
            if filtered and len(filtered) != len(adatas):
                print(f"Skipping {len(adatas) - len(filtered)} empty scPRINT prediction shard(s).")
                adatas = tuple(filtered) if isinstance(adatas, tuple) else filtered
        return original_anndata_concat(adatas, *args, **kwargs)

    ad.concat = concat_without_empty_scprint_shards
    for module_name in ("scprint.model.model", "scprint.tasks.cell_emb"):
        module = importlib.import_module(module_name)
        if getattr(module, "concat", None) is original_anndata_concat:
            module.concat = concat_without_empty_scprint_shards

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
        save_every=${params.scprint_save_every},
        dtype=dtype,
        doplot=False
    )

    import scanpy as sc
    import pandas as pd
    if getattr(sc, "concat", None) is original_anndata_concat:
        sc.concat = concat_without_empty_scprint_shards

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

    publishDir "${params.emb_results_dir}/embeddings/scprint", mode: 'copy',
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


params.finetune_batch_size   = 32
params.finetune_epoch        = 10
params.finetune_eval_size    = 0.2
params.predict_batch_size    = 64
params.finetune_results_dir  = ""

process preprocess_for_scprint_with_ct {

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
    adata = sc.read_h5ad("${raw_h5ad}", backed="r")
    species = "${species}"
    if species.lower() == "human":
        organism_ontology_term_id = "NCBITaxon:9606"

    import pandas as pd
    empty_obs = pd.DataFrame(index=adata.obs_names.tolist())
    empty_obs['barcode'] = adata.obs['barcode'].tolist()
    if "${params.finetune_label_key}" in adata.obs.columns:
        empty_obs['cell_type'] = adata.obs['${params.finetune_label_key}'].tolist()
    adata.obs = empty_obs

    adata.obs['organism_ontology_term_id'] = organism_ontology_term_id
    adata.var_names = adata.var['ensembl_id']

    # if "is_primary_data" in adata.obs.columns:
    #     adata.obs.drop(columns="is_primary_data", inplace=True)
    
    adata.write_h5ad("temp_scprint.h5ad")
    """

}


process _finetune_by_scprint {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/scprint:latest'

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models",
               saveAs: { filename -> "scPRINT/${id}" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad), path(processed_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    source /.venv/bin/activate

    mkdir -p "\$HOME/.lamin"
    cp -a "\${SCPRINT_LAMIN_SEED:-/opt/scprint/lamin}/.lamin/." "\$HOME/.lamin/"

    python - << 'EOF'
    import os
    import torch
    import torch.nn as nn
    from scprint import scPrint
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.callbacks import EarlyStopping
    from scdataloader import Collator, Preprocessor
    from scdataloader.data import SimpleAnnDataset
    from scdataloader.utils import load_genes
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    import scanpy as sc
    import pandas as pd
    import numpy as np
    import types
    from scprint.model import loss
    from typing import Dict, Optional
    from torch import Tensor

    # =========== load data ============ #

    adata = sc.read_h5ad("${processed_h5ad}")
    preprocessor = Preprocessor(
        filter_gene_by_counts = False,
        filter_cell_by_counts = False,
        min_valid_genes_id = 50,
        min_dataset_size = 0,
        min_nnz_genes = 0,
        do_postp=False, skip_validate=True)
    adata = preprocessor(adata)

    labels = np.unique(adata.obs['cell_type'])
    label2id = {item:i for i,item in enumerate(labels)}
    id2label = {v:k for k, v in label2id.items()}
    adata.obs.cell_type = adata.obs.cell_type.map(label2id)

    adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id", "cell_type"])
    labels_list = [adataset[i]['cell_type'] for i in range(len(adataset))]

    # Crash-safe stratified split: every class keeps >=1 training sample; a class too
    # small to spare one for validation (floor(count*eval_size)==0) goes entirely to train.
    def _min_train_split(_labels, _vf, _seed=42):
        _labels = np.asarray(_labels); _rng = np.random.RandomState(_seed)
        _tr, _va = [], []
        for _c in np.unique(_labels):
            _ix = np.where(_labels == _c)[0]; _rng.shuffle(_ix)
            _nv = min(int(len(_ix) * _vf), len(_ix) - 1)
            _va.extend(_ix[:_nv].tolist()); _tr.extend(_ix[_nv:].tolist())
        _rng.shuffle(_tr); _rng.shuffle(_va)
        return np.array(_tr, dtype=int), np.array(_va, dtype=int)
    train_idx, val_idx = _min_train_split(labels_list, ${params.finetune_eval_size}, 42)

    train_dataset = Subset(adataset, train_idx)
    val_dataset = Subset(adataset, val_idx)
    

    # =========== load model ============ #

    model_checkpoint_file = os.path.join("/data/model_weights", "${params.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = torch.load(model_checkpoint_file, map_location=device)
    transformer = "normal" if device.type == "cpu" else "flash"

    if "prenorm" in m["hyper_parameters"]:
        # `prenorm` is a legacy hyper-parameter that scPrint.__init__ no longer accepts
        # (pre-norm is the fixed default). Drop it from a TASK-LOCAL copy of the
        # checkpoint; never rewrite the shared file under /data/model_weights.
        m["hyper_parameters"].pop("prenorm")
        model_checkpoint_file = os.path.abspath("scprint_checkpoint_local.ckpt")
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

    # this might happen if you have a model that was trained with a different set of genes than the one you are using in the ontology (e.g. newer ontologies), While having genes in the onlogy not in the model is fine. the opposite is not, so we need to remove the genes that are in the model but not in the ontology
    missing = set(model.genes) - set(load_genes(model.organisms).index)
    if len(missing) > 0:
        print(
            "Warning: some genes missmatch exist between model and ontology: solving...",
        )
        model._rm_genes(missing)

    # again if not on GPU you need to convert the model to float64
    if device.type == "cpu":
        model = model.to(torch.float32)

    # you can perform your inference on float16 if you have a GPU, otherwise use float64
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    
    # the models are often loaded with some parts still displayed as "cuda" and some as "cpu", so we need to make sure that the model is fully on the right device
    model = model.to(device)

    # =========== change layer ============ #

    in_features = 256
    out_features = len(labels)
    new_out_layer = torch.nn.Linear(in_features, out_features)
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    new_out_layer.apply(_init)
    new_out_layer.to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    model.cls_decoders.cell_type_ontology_term_id.out_layer = new_out_layer
    clsname = "cell_type_ontology_term_id"
    model.label_counts[clsname] = out_features

    # =========== change function =========== #

    def _full_training(
        self,
        batch: Dict[str, Tensor],
        do_denoise: bool = False,
        noise: list[float] = [],
        do_next_tp: bool = False,
        do_cce: bool = False,
        cce_temp: float = 0.5,
        do_ecs: bool = False,
        do_mvc: bool = False,
        do_adv_cls: bool = False,
        do_adv_batch: bool = False,
        do_cls: bool = False,
        do_generate: bool = False,
        run_full_forward: bool = True,
        mask_ratio: list[float] = [0.15],
    ):
        device = self.device.type
        gene_pos, expression, depth, clss = (
            batch["genes"].to(device),
            batch["x"].to(device),
            batch["depth"].to(device),
            batch["class"].to(device),
        )
            
        output = self.forward(
            gene_pos,
            expression,
            mask=None,
            req_depth=depth,
            do_mvc=self.do_mvc,
            do_class=True,
            metacell_token=None,
        )
            
        total_loss = 0
        losses = {}
        
        loss_emb_indep = loss.within_sample(output["cell_embs"])
        total_loss += self.class_embd_diss_scale * loss_emb_indep

        loss_cls = 0
        loss_adv_cls = 0
        clsname = "cell_type_ontology_term_id"
        loss_cls += loss.classification(
            clsname,
            pred=output["cls_output_" + clsname],
            cl=clss.flatten(),
            maxsize=self.label_counts[clsname],
            labels_hierarchy=[],
        )
        total_loss += self.class_scale * loss_cls
        losses['train_loss'] = total_loss
        return total_loss, losses

    def on_validation_epoch_end(self):
        \"\"\"@see pl.LightningModule\"\"\"
        self.embs = self.all_gather(self.embs).view(-1, self.embs.shape[-1])
        self.info = self.all_gather(self.info).view(-1, self.info.shape[-1])
        self.pred = (
            self.all_gather(self.pred).view(-1, self.pred.shape[-1])
            if self.pred is not None
            else None
        )
        self.pos = self.all_gather(self.pos).view(-1, self.pos.shape[-1])
        if self.trainer.state.stage != "sanity_check":
            if self.trainer.is_global_zero:
                sch = self.lr_schedulers()
                sch.step(self.trainer.callback_metrics["val_loss"])
                # run the test function on specific dataset
                if (self.current_epoch + 1) % self.test_every == 0:
                    self.on_test_epoch_end()
                # Synchronize all processes with a timeout
            if torch.distributed.is_initialized():
                # Set a timeout that's longer than your test typically takes
                # Write rank to file for debugging
                self.trainer.strategy.barrier()

    model._full_training = types.MethodType(_full_training, model)
    model.pred_log_adata = False
    model.on_validation_epoch_end = types.MethodType(on_validation_epoch_end, model)
    model.gene_encoder.embeddings.requires_grad_(False)
    
    col = Collator(
        organisms=model.organisms,
        valid_genes=model.genes,
        how="random expr",
        max_len=4000,
        add_zero_genes=0,
        genelist=[],
        class_names=['cell_type']
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=col,
        batch_size=${params.finetune_batch_size},
        num_workers=16,
        shuffle=True,
    )

    valid_dataloader = DataLoader(
        val_dataset,
        collate_fn=col,
        batch_size=2* ${params.finetune_batch_size},
        num_workers=16,
        shuffle=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       # 监控的指标
        mode="min",               # min=越小越好, max=越大越好
        save_top_k=1,             # 保存最优的1个
        save_last= False,
        filename="best-checkpoint"
    )

    earlystop_callback = EarlyStopping(
        monitor='val_loss',
        patience=4,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=${params.finetune_epoch}, 
        precision="16-mixed",
        gradient_clip_val=40,
        log_every_n_steps= 100,
        limit_train_batches= 20000,
        gradient_clip_algorithm= 'norm',
        limit_val_batches= 2000,
        limit_test_batches= 1, # we don't perform tests this way
        reload_dataloaders_every_n_epochs= 5,
        accumulate_grad_batches= 1,
        callbacks = [checkpoint_callback, earlystop_callback]
    )

    trainer.fit(model, train_dataloader, val_dataloaders=valid_dataloader)

    # ========== save model ========== #

    import json
    from pathlib import Path

    best_path = checkpoint_callback.best_model_path
    ckpt = torch.load(best_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    save_dir = Path("./scprint_finetuned_model")
    save_dir.mkdir()
    torch.save(model, save_dir/"best-model.pth")

    # import shutil
    # import glob
    # best_checkpoint = checkpoint_callback.best_model_path
    # save_dir = Path("./scprint_finetuned_model")
    # save_dir.mkdir()
    # shutil.move(best_checkpoint, save_dir / Path(best_checkpoint).name)

    with open(save_dir / "label_map.json", "w") as f:
        json.dump(
            {
                "id2label": id2label,
                "label2id": label2id,
            },
            f
        )

    EOF
    """
}



process _predict_by_scprint {

    tag "${id}"

    label 'gpu_task'

    container 'housy17/scprint:latest'

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "scPRINT/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(test_h5ad), path(processed_test_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    source /.venv/bin/activate

    mkdir -p "\$HOME/.lamin"
    cp -a "\${SCPRINT_LAMIN_SEED:-/opt/scprint/lamin}/.lamin/." "\$HOME/.lamin/"

    python - << 'EOF'
    #!/usr/bin/env python
    import os
    import torch
    import torch.nn as nn
    from scprint import scPrint
    from scdataloader import Collator, Preprocessor
    from scdataloader.data import SimpleAnnDataset
    from scdataloader.utils import load_genes
    from torch.utils.data import DataLoader

    import scanpy as sc
    import pandas as pd
    import numpy as np
    import types
    from scprint.model import loss
    from typing import Dict, Optional
    from torch import Tensor
    from pathlib import Path
    import json
    from tqdm import tqdm
    

    # =========== load data ============ #

    adata = sc.read_h5ad("${processed_test_h5ad}")
    preprocessor = Preprocessor(
        filter_gene_by_counts = False,
        filter_cell_by_counts = False,
        min_valid_genes_id = 50,
        min_dataset_size = 0,
        min_nnz_genes = 0,
        do_postp=False, skip_validate=True)
    adata = preprocessor(adata)

    model_dir = Path("${model_weights}")
    label_map_file = model_dir / "label_map.json"
    with open(label_map_file, "r") as f:
        tmp = json.load(f)
        id2label = {int(k):v for k, v in tmp["id2label"].items()}
        label2id = tmp["label2id"]

    adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])

    # =========== load model ============ #

    # model_checkpoint_file = model_dir / "best-checkpoint.ckpt"
    model = torch.load(model_dir / "best-model.pth")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        model = model.to(torch.float32)
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    model = model.to(device)

    # =========== data loader ============= #

    col = Collator(
        organisms=model.organisms,
        valid_genes=model.genes,
        how="random expr",
        max_len=4000,
        add_zero_genes=0,
        genelist=[],
    )

    eval_batch_size = ${params.predict_batch_size}
    test_dataloader = DataLoader(
        adataset,
        collate_fn=col,
        batch_size=eval_batch_size,
        num_workers=16,
        shuffle=False,
    )

    model.eval()
    model.on_predict_epoch_start()

    pred_proba = np.zeros((len(adata), len(label2id)))

    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=dtype),
    ):
        for i, batch in tqdm(enumerate(test_dataloader)):
            gene_pos, expression, depth = (
                batch["genes"].to(device),
                batch["x"].to(device),
                batch["depth"].to(device),
            )

            output = model.forward(
                gene_pos,
                expression,
                depth_mult=expression.sum(1),
                req_depth=depth,
                get_attention_layer=[],
                do_class=True,
                get_gene_emb=False,
                metacell_token=None,
            )

            output = output['cls_output_cell_type_ontology_term_id']
            pred_proba[i*eval_batch_size:(i+1)*eval_batch_size] = output.cpu().numpy()
            torch.cuda.empty_cache()

    from scipy.special import softmax
    pred_proba = softmax(pred_proba, axis=-1)
    label_names = [id2label[i] for i in range(len(id2label))]
    barcodes = adata.obs.barcode.tolist()
    import pandas as pd
    df = pd.DataFrame(pred_proba, columns=label_names, index=barcodes)
    df.to_csv("scprint_predictions.tsv", sep="\\t")

    EOF
    """
}
