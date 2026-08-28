params.model = "CellFM/CellFM_80M_weight.ckpt"
params.batch_size = 64
params.emb_results_dir = "results"

process embed_by_cellfm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellfm:latest"

    publishDir "${params.emb_results_dir}/embeddings/cellfm", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("CellFM"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import sys
    sys.path.append("/code/cellfm")
    import scanpy as sc
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.cuda.amp import autocast, GradScaler
    from layers.utils import *
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from model import Cell_FM
    import os
    import anndata
    from scipy.sparse import csr_matrix, issparse

    import warnings
    warnings.filterwarnings("ignore")

    def apply_cellfm_count_cap(adata, max_counts=1e5):
        X = adata.X.astype(np.float32)
        totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.maximum(1.0, totals / max_counts).astype(np.float32)
        if issparse(X):
            X = X.tocsr(copy=True)
            X = X.multiply((1.0 / scale)[:, None]).tocsr()
            X.data = np.round(X.data).astype(np.float32)
        else:
            X = csr_matrix(np.round(X / scale[:, None]).astype(np.float32))
        X.eliminate_zeros()
        adata.X = X

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_path = os.path.join("/data/model_weights", "${params.model}")

    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = ${params.batch_size}
    cfg.mask_ratio = 0.0 # no masked reconstruction for zero-shot embedding
    cfg.device = device
    cfg.ckpt_path = ckpt_path

    model = Cell_FM(27855, cfg, ckpt_path=cfg.ckpt_path, device=cfg.device)
    model = model.to(cfg.device)
    model.load_model(weight=True, moment=False)

    adata = sc.read_h5ad("${raw_h5ad}")
    apply_cellfm_count_cap(adata)
    # cellfm has .toarray() in its code, therefore csr data is needed.
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    elif not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    obs = adata.obs.copy()
    empty_obs = pd.DataFrame(index=adata.obs_names.tolist())
    empty_obs['barcode'] = adata.obs['barcode'].tolist()
    adata.obs = empty_obs
    adata.var_names_make_unique()
    adata.obs['train'] = 2
    
    # Emmm — although no cell type info is used in embedding process, it still need this to build dataloader.
    if "celltype" not in adata.obs.columns:
        adata.obs["celltype"] = ["virtual_ct"] * len(adata)

    # Even bugs in original codes!
    if "batch_id" not in adata.obs.columns:
        adata.obs["batch_id"] = [0] * len(adata)

    # No features info in zero-shot.
    adata.obs["feat"] = [0.0] * len(adata)

    dataset = SCrna(adata, mode="test")
    # X, gene, T: count sum, celltype_label, batch_id, feat

    # With the full-panel fix, cells with >2048 nonzero genes are expression-weighted
    # subsampled in Prepare.sample. Seed numpy so embeddings are reproducible.
    np.random.seed(0)

    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio, dw=False)
    loader = build_dataset(
        dataset,
        prep=prep,
        batch_size=cfg.use_bs,
        drop=False,
        shuffle=False
    )

    embedding_list = []
    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(loader):    
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)

            emb, _ = model.net.encode(
                raw_nzdata,
                nonz_gene,
                ST_feat,
                zero_idx
            )
            gene_tokens = emb[:, 3:]
            gene_mask = zero_idx.unsqueeze(-1)
            cell_embedding = (gene_tokens * gene_mask).sum(dim=1) / gene_mask.sum(dim=1).clamp_min(1.0)

            embedding_list.append(cell_embedding.detach().cpu().numpy())
    embedding = np.vstack(embedding_list)
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = anndata.AnnData(X=embedding, obs=obs, var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write("cellfm_embeddings.h5ad")
    """
}


params.finetune_epoch        = 30   // MindSpore official count (we use the MindSpore annotation head)
params.finetune_eval_size    = 0.2
params.finetune_batch_size   = 8
params.predict_batch_size    = 64
params.finetune_results_dir  = ""

process finetune_by_cellfm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellfm:latest"

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models",
               saveAs: { filename -> "CellFM/${id}" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    #!/usr/bin/env python
    import sys
    sys.path.append("/code/cellfm")
    import scanpy as sc
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.cuda.amp import autocast, GradScaler
    from layers.utils import *
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from model import Cell_FM
    import os
    import anndata
    from scipy.sparse import csr_matrix, issparse
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    save_dir = Path("cellfm_finetuned_model/")
    save_dir.mkdir()
    import json
    import warnings
    warnings.filterwarnings("ignore")

    def apply_cellfm_count_cap(adata, max_counts=1e5):
        X = adata.X.astype(np.float32)
        totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.maximum(1.0, totals / max_counts).astype(np.float32)
        if issparse(X):
            X = X.tocsr(copy=True)
            X = X.multiply((1.0 / scale)[:, None]).tocsr()
            X.data = np.round(X.data).astype(np.float32)
        else:
            X = csr_matrix(np.round(X / scale[:, None]).astype(np.float32))
        X.eliminate_zeros()
        adata.X = X

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_path = os.path.join("/data/model_weights", "${params.model}")

    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = ${params.finetune_batch_size} # batch_size
    cfg.mask_ratio = 0.0  # MindSpore CellFM annotation uses no masking (train or infer)
    cfg.device = device
    cfg.ckpt_path = ckpt_path
    cfg.feature_col = "${params.finetune_label_key}"
    cfg.epoch = ${params.finetune_epoch}

    adata = sc.read_h5ad("${raw_h5ad}")
    apply_cellfm_count_cap(adata)
    # cellfm has .toarray() in its code, therefore csr data is needed.
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    elif not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    adata.obs[cfg.feature_col] = adata.obs[cfg.feature_col].astype("category")
    adata.obs['celltype'] = adata.obs["${params.finetune_label_key}"]
    adata.obs['feat'] = adata.obs[cfg.feature_col].cat.codes.values

    labels = list(adata.obs[cfg.feature_col].cat.categories)   # label 的顺序 = id 对应顺序
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    with open(save_dir / "label_map.json", "w") as f:
        json.dump(
            {
                "id2label": id2label,
                "label2id": label2id,
            },
            f
        )

    cfg.num_cls = len(adata.obs['feat'].unique())
    adata.obs['batch_id'] = 0
    adata.obs['train'] = 0
    label_list = adata.obs[cfg.feature_col].tolist()
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
    train_idx, val_idx = _min_train_split(label_list, ${params.finetune_eval_size}, 42)
    adata.obs.iloc[val_idx, adata.obs.columns.get_loc("train")] = 1

    np.random.seed(0)
    torch.manual_seed(0)

    train_dataset = SCrna(adata, mode="train")
    eval_dataset = SCrna(adata, mode="val")
    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
    train_loader = build_dataset(
        train_dataset,
        prep=prep,
        batch_size=cfg.use_bs,
        drop=True,
        shuffle=True
    )

    eval_loader = build_dataset(
        eval_dataset,
        prep=prep,
        batch_size=2 * cfg.use_bs,
        drop=False,
        shuffle=False
    )

    from annotation import Annotation_Cell_FM

    # class weights (inverse frequency) over the training split, MindSpore-style
    train_feat = adata.obs.loc[adata.obs['train'] == 0, 'feat'].values.astype(int)
    counts = np.bincount(train_feat, minlength=cfg.num_cls).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weight = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=cfg.device)

    # MindSpore-style annotation head: cluster prototypes + cross-attention + dual logits.
    net = Annotation_Cell_FM(cfg)
    for name, param in net.named_parameters():
        param.requires_grad = "head." in name or "encoder" in name

    print("Trainable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    net = net.to(cfg.device)
    net.extractor.load_model(weight=True, moment=False)

    optimizer = AdamW([p for p in net.parameters() if p.requires_grad], 
                    lr=1e-4,
                    weight_decay=1e-5)

    scaler = GradScaler() 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weight)

    best_eval_loss = np.inf
    for epoch in range(cfg.epoch):
        net.train()
        print("training...")
        running_loss = 0.0
        running_acc = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epoch}")

        for step, batch in enumerate(progress):    
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)
            celltype_label = batch['celltype_label'].to(cfg.device)
            batch_id = batch['batch_id'].to(cfg.device)
            feat = batch['feat'].long().to(cfg.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                lp1, lp2 = net(
                    raw_nzdata=raw_nzdata,
                    dw_nzdata=dw_nzdata,
                    ST_feat=ST_feat,
                    nonz_gene=nonz_gene,
                    mask_gene=mask_gene,
                    zero_idx=zero_idx
                )
                loss = criterion_cls(lp1, feat) + criterion_cls(lp2, feat)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            pred = (F.softmax(lp1, dim=1) + F.softmax(lp2, dim=1)).argmax(1)
            accuracy = (pred == feat).sum().item()
            accuracy = accuracy / len(batch_id)

            running_loss += loss.item()
            running_acc += accuracy
            
            avg_loss = running_loss / (step + 1)
            avg_acc = running_acc / (step + 1)
            progress.set_postfix(loss=avg_loss, acc=avg_acc)

        scheduler.step()
        net.eval()
        print("testing...")
        running_loss = 0.0
        running_acc = 0.0
        progress = tqdm(eval_loader, desc="Evaluating")
        with torch.no_grad(): 
            for step, batch in enumerate(progress): 
                raw_nzdata = batch['raw_nzdata'].to(cfg.device)
                dw_nzdata = batch['dw_nzdata'].to(cfg.device)
                ST_feat = batch['ST_feat'].to(cfg.device)
                nonz_gene = batch['nonz_gene'].to(cfg.device)
                mask_gene = batch['mask_gene'].to(cfg.device)
                zero_idx = batch['zero_idx'].to(cfg.device)
                celltype_label = batch['celltype_label'].to(cfg.device)
                batch_id = batch['batch_id'].to(cfg.device)
                feat = batch['feat'].long().to(cfg.device)
            
                with torch.cuda.amp.autocast():
                    lp1, lp2 = net(
                        raw_nzdata=raw_nzdata,
                        dw_nzdata=dw_nzdata,
                        ST_feat=ST_feat,
                        nonz_gene=nonz_gene,
                        mask_gene=mask_gene,
                        zero_idx=zero_idx
                    ) 

                    loss = criterion_cls(lp1, feat) + criterion_cls(lp2, feat)

                pred = (F.softmax(lp1, dim=1) + F.softmax(lp2, dim=1)).argmax(1)
                accuracy = (pred == feat).sum().item()
                accuracy = accuracy / len(batch_id)

                running_loss += loss.item()
                running_acc += accuracy
                avg_loss = running_loss / (step + 1)
                avg_acc = running_acc / (step + 1)
                
                progress.set_postfix(loss=avg_loss, acc=avg_acc)
        print(f"Testing {epoch+1} complete,avg loss: {avg_loss:.6f}, avg acc: {avg_acc:.6f}")
        if avg_loss < best_eval_loss:
            best_eval_loss = avg_loss
            torch.save(net.state_dict(), f"cellfm_finetuned_model/best.pth")

    """
}



process predict_by_cellfm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellfm:latest"

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "CellFM/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(test_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    #!/usr/bin/env python
    import sys
    sys.path.append("/code/cellfm")
    import scanpy as sc
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.cuda.amp import autocast, GradScaler
    from layers.utils import *
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from model import Cell_FM
    import os
    import anndata
    from scipy.sparse import csr_matrix, issparse
    from pathlib import Path
    import json

    import warnings
    warnings.filterwarnings("ignore")

    def apply_cellfm_count_cap(adata, max_counts=1e5):
        X = adata.X.astype(np.float32)
        totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.maximum(1.0, totals / max_counts).astype(np.float32)
        if issparse(X):
            X = X.tocsr(copy=True)
            X = X.multiply((1.0 / scale)[:, None]).tocsr()
            X.data = np.round(X.data).astype(np.float32)
        else:
            X = csr_matrix(np.round(X / scale[:, None]).astype(np.float32))
        X.eliminate_zeros()
        adata.X = X

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join("${model_weights}", "best.pth")

    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = ${params.predict_batch_size}
    cfg.mask_ratio = 0.0  # no masking at inference (MindSpore CellFM annotation)
    cfg.device = device
    cfg.ckpt_path = ckpt_path

    label_map_file = os.path.join("${model_weights}", "label_map.json")
    with open(label_map_file, "r") as f:
        tmp = json.load(f)
        id2label = {int(k):v for k, v in tmp["id2label"].items()}
        label2id = tmp["label2id"]
    cfg.num_cls = len(label2id)

    from annotation import Annotation_Cell_FM
    # MindSpore-style annotation head.
    model = Annotation_Cell_FM(cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    adata = sc.read_h5ad("${test_h5ad}")
    apply_cellfm_count_cap(adata)
    # cellfm has .toarray() in its code, therefore csr data is needed.
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    elif not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    obs = adata.obs.copy()
    empty_obs = pd.DataFrame(index=adata.obs_names.tolist())
    empty_obs['barcode'] = adata.obs['barcode'].tolist()
    adata.obs = empty_obs
    adata.var_names_make_unique()
    adata.obs['train'] = 2
    adata.obs["celltype"] = ["virtual_ct"] * len(adata)
    adata.obs["batch_id"] = [0] * len(adata)
    adata.obs["feat"] = [0] * len(adata)

    np.random.seed(0)
    dataset = SCrna(adata, mode="test")
    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio, dw=False)
    loader = build_dataset(
        dataset,
        prep=prep,
        batch_size=cfg.use_bs,
        drop=False,
        shuffle=False
    )

    embedding_list = []
    model.eval()
    probs = []
    with torch.inference_mode():
        for step, batch in enumerate(loader):    
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)
            
            lp1, lp2 = model(
                raw_nzdata,
                dw_nzdata,
                ST_feat,
                nonz_gene,
                mask_gene,
                zero_idx
            )
            probs.append(Annotation_Cell_FM.predict_proba(lp1, lp2))
    pred_proba = torch.cat(probs)
    pred_proba = pred_proba.cpu().numpy()

    label_names = [id2label[i] for i in range(len(id2label))]
    barcodes = adata.obs.barcode.tolist()
    import pandas as pd
    df = pd.DataFrame(pred_proba, columns=label_names, index=barcodes)
    df.to_csv("cellfm_predictions.tsv", sep="\\t")
    """

}
