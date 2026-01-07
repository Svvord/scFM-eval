params.batch_size = 64
params.results_dir = "results"

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

    publishDir "${params.results_dir}/embeddings/langcell", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" }

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
