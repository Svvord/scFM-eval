params.model = "CellFM/CellFM_80M_weight.ckpt"
params.batch_size = 64
params.results_dir = "results"

process embed_by_cellfm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellfm:latest"

    publishDir "${params.results_dir}/embeddings/cellfm", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_path = os.path.join("/data/model_weights", "${params.model}")

    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = ${params.batch_size}
    cfg.mask_ratio = 0.0 # set mask ratio as zero
    cfg.device = device
    cfg.ckpt_path = ckpt_path

    model = Cell_FM(27855, cfg, ckpt_path=cfg.ckpt_path, device=cfg.device)
    model = model.to(cfg.device)

    adata = sc.read_h5ad("${raw_h5ad}")
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
    # X, gene, T: count sum, celltypel_label, batch_id, feat

    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
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
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)
            
            # prepare.mask 代码就是错的, 需要绕过这个代码
            mask_gene = torch.zeros_like(mask_gene)

            _, cls_token = model(
                raw_nzdata,
                raw_nzdata, #dw_nzdata, # 实际测试应该给 raw_nzdata
                ST_feat,
                nonz_gene,
                mask_gene,
                zero_idx
            )

            embedding_list.append(cls_token.detach().cpu().numpy())
    embedding = np.vstack(embedding_list)
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = anndata.AnnData(X=embedding, obs=obs, var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write("cellfm_embeddings.h5ad")
    """
}