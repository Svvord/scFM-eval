params.model = "CellPLM/20231027_85M.best.ckpt"
params.batch_size = 1024   // inference batch size used for the manuscript runs
params.emb_results_dir = "results"

process embed_by_cellplm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellplm:latest"

    publishDir "${params.emb_results_dir}/embeddings/cellplm", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }, enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad) // 内部转成 log1p transformed

    output:
    tuple val(id), val("CellPLM"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import warnings
    warnings.filterwarnings("ignore")
    
    import hdf5plugin
    import numpy as np
    import anndata as ad
    from scipy.sparse import csr_matrix
    from CellPLM.utils import set_seed
    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
    import scanpy as sc
    import matplotlib.pyplot as plt
    import rapids_singlecell as rsc
    import os
    import torch
    import anndata
    import pandas as pd


    PRETRAIN_VERSION = os.path.basename("${params.model}").split('.')[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    adata = sc.read_h5ad("${raw_h5ad}")
    adata.var_names = adata.var['ensembl_id'].tolist()
    adata.var_names_make_unique()

    PRETRAIN_DIR = os.path.join("/data/model_weights", "CellPLM")
    pipeline = CellEmbeddingPipeline(
        pretrain_prefix=PRETRAIN_VERSION,
        pretrain_directory=PRETRAIN_DIR
    )
    embedding = pipeline.predict(
        adata, # An AnnData object
        inference_config={'batch_size': ${params.batch_size}},
        device=device
    )
    embedding = embedding.detach().cpu().numpy()
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    adata_embedding = anndata.AnnData(X=embedding, obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in adata.obsm:
        adata_embedding.obsm['spatial'] = adata.obsm['spatial']
    adata_embedding.write("cellplm_embeddings.h5ad")
    """
    
}


params.finetune_epoch        = 2000   // deafult: 2000
params.finetune_eval_size    = 0.2
// params.cellplm_finetune_batch_size   = 32   // 没有意义
// params.cellplm_predict_batch_size    = 64   // 没有意义
params.finetune_results_dir  = ""


process finetune_by_cellplm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellplm:latest"

    publishDir "${params.finetune_results_dir}/finetune/finetuned_models", mode: 'copy',
               saveAs: { filename -> "cellplm/${id}" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), path("*_finetuned_model")

    script:
    """
    #!/usr/bin/env python
    import warnings
    warnings.filterwarnings("ignore")
    
    import hdf5plugin
    import numpy as np
    import anndata as ad
    from scipy.sparse import issparse
    from scipy.sparse import csr_matrix
    from CellPLM.utils import set_seed
    from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationPipeline, CellTypeAnnotationDefaultPipelineConfig, CellTypeAnnotationDefaultModelConfig
    from sklearn.model_selection import train_test_split
    import scanpy as sc
    import os
    import json
    import torch

    PRETRAIN_DIR = os.path.join("/data/model_weights", "CellPLM")
    PRETRAIN_VERSION = os.path.basename("${params.model}").split('.')[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    adata = sc.read_h5ad("${raw_h5ad}")
    adata.var_names = adata.var['ensembl_id'].tolist()
    adata.var_names_make_unique()

    adata.obs['cell_type'] = adata.obs['${params.finetune_label_key}'].tolist()
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

    labels_list = adata.obs['cell_type'].tolist()
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    
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

    adata.obs['split'] = 'train'
    adata.obs.iloc[val_idx, adata.obs.columns.get_loc('split')] = 'valid'

    pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
    pipeline_config['epochs'] = ${params.finetune_epoch}
    # default config is:
    # 'es': 200,  early_stop 的意思
    # 'lr': 5e-3,
    # 'wd': 1e-7,
    # 'scheduler': 'plat',
    # 'epochs': 2000,
    # 'max_eval_batch_size': 100000,
    # 'hvg': 3000,
    # 'patience': 25,
    # 'workers': 0,

    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    model_config['out_dim'] = len(np.unique(labels_list))
    # 'drop_node_rate': 0.3,
    # 'dec_layers': 1,
    # 'model_dropout': 0.5,
    # 'mask_node_rate': 0.75,
    # 'mask_feature_rate': 0.25,
    # 'dec_mod': 'mlp',
    # 'latent_mod': 'ae',
    # 'head_type': 'annotation',
    # 'max_batch_size': 70000,
    
    print(pipeline_config)
    print(model_config)

    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
        overwrite_config=model_config,  # This is for overwriting part of the pretrain config
        pretrain_directory=PRETRAIN_DIR,
    )

    pipeline.fit(
        adata, # An AnnData object
        pipeline_config, # The config dictionary we created previously, optional
        split_field = 'split', #  Specify a column in .obs that contains split information
        train_split = 'train',
        valid_split = 'valid',
        label_fields = ['cell_type'],
        device = device
    )

    def save_pretrain(model, config: dict, pretrain_prefix: str,
                  pretrain_directory: str = './cellplm_finetuned_model'):
        \"\"\"
        保存 finetune 后的模型，使其能被 load_pretrain() 正常加载
        \"\"\"

        os.makedirs(pretrain_directory, exist_ok=True)

        config_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.config.json')
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        ckpt_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.best.ckpt')
        torch.save(
            {
                'model_state_dict': model.state_dict()
            },
            ckpt_path
        )
        print(f"Config saved to: {config_path}")
        print(f"Checkpoint saved to: {ckpt_path}")

    with open(os.path.join(PRETRAIN_DIR, PRETRAIN_VERSION+'.config.json'), "r") as openfile:
        config = json.load(openfile)
    config.update(model_config)

    save_pretrain(pipeline.model, config, pretrain_prefix='finetuned')

    
    import pickle
    with open(os.path.join('cellplm_finetuned_model', 'finetuned.label_encoders.pkl'), 'wb') as f:
        pickle.dump(pipeline.label_encoders, f)
    # 越过 common preprocess 的 fitted 检查, 要保存 pipeline 的 gene_list
    with open(os.path.join('cellplm_finetuned_model', 'finetuned.gene_list.pkl'), 'wb') as f:
        pickle.dump(pipeline.gene_list, f)
    """

}



process predict_by_cellplm {

    tag "${id}"

    label "gpu_task"

    container "housy17/cellplm:latest"

    publishDir "${params.finetune_results_dir}/finetune/prediction", mode: 'copy', pattern: "*_predictions.tsv", 
               saveAs: { filename -> "cellplm/${id}_predicted_probs.tsv" }, enabled: params.finetune_results_dir as boolean

    input:
    tuple val(id), path(test_h5ad), path(model_weights)

    output:
    tuple val(id), path("*_predictions.tsv")

    script:
    """
    #!/usr/bin/env python
    import warnings
    warnings.filterwarnings("ignore")
    
    import numpy as np
    import pickle
    from scipy.sparse import issparse
    from scipy.sparse import csr_matrix
    from CellPLM.utils import set_seed
    from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationPipeline, CellTypeAnnotationDefaultPipelineConfig, CellTypeAnnotationDefaultModelConfig
    from CellPLM.utils.data import XDict, TranscriptomicDataset

    import scanpy as sc
    import os
    import json
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F


    def inference(model, dataloader, split, device, batch_size, eval_dict, label_fields=None, order_required=False):
        if order_required and split:
            warnings.warn('When cell order required to be preserved, dataset split will be ignored.')

        with torch.no_grad():
            model.eval()
            epoch_loss = []
            order_list = []
            probs = []

            for i, data_dict in enumerate(dataloader):
                if not order_required and split and np.sum(data_dict['split'] == split) == 0:
                    continue

                idx = torch.arange(data_dict['x_seq'].shape[0])
                if split:
                    data_dict['loss_mask'] = torch.from_numpy((data_dict['split'] == split).values).bool()
                else:
                    data_dict['loss_mask'] = torch.ones(data_dict['x_seq'].shape[0]).bool()
                if label_fields:
                    data_dict['label'] = data_dict[label_fields[0]]
                for j in range(0, len(idx), batch_size):
                    if len(idx) - j < batch_size:
                        cur = idx[j:]
                    else:
                        cur = idx[j:j + batch_size]
                    input_dict = {}
                    for k in data_dict:
                        if k =='x_seq':
                            input_dict[k] = data_dict[k].index_select(0, cur).to(device)
                        elif k not in ['gene_list', 'split']:
                            input_dict[k] = data_dict[k][cur].to(device)
                    x_dict = XDict(input_dict)
                    out_dict, loss = model(x_dict, data_dict['gene_list'])
                    if order_required:
                        order_list.append(input_dict['order_list'])
                    probs.append(out_dict['probs'])

            probs = torch.cat(probs)
            if order_required:
                order = torch.cat(order_list)
                order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
                probs = probs[order]

        return probs

    # ============ load data ============= #
    adata = sc.read_h5ad("${test_h5ad}")
    adata.var_names = adata.var['ensembl_id'].tolist()
    adata.var_names_make_unique()
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)

    
    # ======= load pipeline data ========= #
    PRETRAIN_DIR = "${model_weights}"
    PRETRAIN_VERSION = "finetuned"
    with open(os.path.join(PRETRAIN_DIR, PRETRAIN_VERSION+'.config.json'), "r") as openfile:
        config = json.load(openfile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
        overwrite_config={"out_dim": config["out_dim"]}, # 强制检查, 所以手工给入
        pretrain_directory=PRETRAIN_DIR
    )
    
    # cellplm 拒绝 cell type annotation zero-shot, 绕过它
    pipeline.fitted = True 
    with open(os.path.join(PRETRAIN_DIR, PRETRAIN_VERSION+'.gene_list.pkl'), 'rb') as f:
        pipeline.gene_list = pickle.load(f)
    with open(os.path.join(PRETRAIN_DIR, PRETRAIN_VERSION+'.label_encoders.pkl'), 'rb') as f:
        pipeline.label_encoders = pickle.load(f)

    label2id = {label: i for i, label in enumerate(pipeline.label_encoders['cell_type'].classes_)}
    id2label = {v:k for k, v in label2id.items()}

    
    inference_config = CellTypeAnnotationDefaultPipelineConfig.copy()
    inference_config.update(pipeline_config)
    pipeline.model.to(device)
    adata = pipeline.common_preprocess(adata, inference_config['hvg'], None, True)
    print(f'After filtering, {adata.shape[1]} genes remain.')
    dataset = TranscriptomicDataset(adata, None, None, order_required=True)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=inference_config['workers'])
    
    # Hou, S. Updated: 替换forward, 把logits暴露出来
    def forward(self, x_dict):
        logits = self.mlp(x_dict['h'][x_dict['loss_mask']])
        probs = F.softmax(logits, dim=-1)
        return {'probs': probs, 'latent': x_dict['h']}, torch.tensor(float('nan'))

    import types
    pipeline.model.head.forward = types.MethodType(forward, pipeline.model.head)

    pred_proba = inference(pipeline.model, dataloader, None, device,
                inference_config['max_eval_batch_size'], 
                pipeline.eval_dict, order_required=True)
    pred_proba = pred_proba.cpu().numpy()

    label_names = [id2label[i] for i in range(len(id2label))]
    barcodes = adata.obs.barcode.tolist()
    import pandas as pd
    df = pd.DataFrame(pred_proba, columns=label_names, index=barcodes)
    df.to_csv("cellplm_predictions.tsv", sep="\\t")
    """

}
