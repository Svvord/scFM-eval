# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import logging
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import issparse
import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset

from performer_pytorch import PerformerLM
import scanpy as sc
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=7, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--data_path", type=str, default='./data/data.h5ad', help='Path of data for generating the embeddings.')
parser.add_argument("--model_path", type=str, default='./model.pth', help='Path of model training on the data.')
parser.add_argument("--save_dir", type=str, default='./attention/', help='Directory of embeddings to save.')
parser.add_argument("--batch_size", type=int, default=32, help='Batch size.')
parser.add_argument("--device", type=str, default='gpu', help='gpu or cpu.')


import torch
from contextlib import contextmanager

@contextmanager
def torch_peak_mem(device="cuda"):
    if isinstance(device, str):
        if device == "cuda":
            device = torch.device("cuda:0")
        else:
            device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    else:
        device = torch.device(device)
    
    torch.cuda.set_device(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        yield
    finally:
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        print(f"[Torch] Peak allocated:  {peak_alloc/1024/1024:.2f} MB")
        print(f"[Torch] Peak reserved:   {peak_reserved/1024/1024:.2f} MB")
        with open("gpu_record.txt", "wt") as f:
            f.write("Peak allocated\tPeak reserved\n")
            f.write(f"{peak_alloc/1024/1024:.2f}\t{peak_reserved/1024/1024:.2f}")

with torch_peak_mem("cuda"):

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    SEQ_LEN = args.gene_num + 1
    CLASS = args.bin_num + 2

    data_dir = args.data_path
    model_dir = args.model_path
    save_dir = args.save_dir


    device = args.device
    if device != "cpu":
        assert torch.cuda.is_available(), "GPU is not available."
    if device == "gpu":
        device = "cuda"
    device = torch.device(device)
    print('            =======  Config over  ======= \n')

    data = sc.read_h5ad(data_dir)

    # index_labels = data.obs['celltype']
    # cellinds = list(set(index_labels.tolist()))
    # label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)

    if issparse(data.X):
        data.X = data.X.toarray()
    data_counts = data.X

    class SCDataset(Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def __getitem__(self, index):
            full_seq = self.data[index]
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
            return full_seq

        def __len__(self):
            return self.data.shape[0]

    dataset = SCDataset(data.X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = True
    )
    print(f'            =======  Model defined  ======= \n')

    ckpt = torch.load(model_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print('            =======  Predict start  ======= \n')

    embedding = torch.zeros(len(dataset), 200)
    model.eval()
    with torch.inference_mode():

        for i, mini_batch in enumerate(dataloader):
            
            full_seq = mini_batch.to(device)
            cell_emb = model(full_seq, return_encodings=True)
            cell_emb = cell_emb[:,-1,:]
            embedding[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = cell_emb.cpu()
    embedding = embedding.numpy()
    var_names = [f'V{i+1}' for i in range(embedding.shape[1])]
    import anndata
    import pandas as pd
    adata_embedding = anndata.AnnData(X=embedding, obs=data.obs.copy(), var=pd.DataFrame(index=var_names))
    if 'spatial' in data.obsm:
        adata_embedding.obsm['spatial'] = data.obsm['spatial']
    adata_embedding.write("scbert_embeddings.h5ad")