# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--novel_type", type=bool, default=False, help='Novel cell tpye exists or not.')
parser.add_argument("--unassign_thres", type=float, default=0.5, help='The confidence score threshold for novel cell type annotation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for predicting.')
parser.add_argument("--model_path", type=str, default='./scbert_finetuned_model/', help='Path of finetuned model.')
parser.add_argument("--batch_size", type=int, default=64, help='Number of batch size.')
args = parser.parse_args()

model_dir = Path(args.model_path)
label_map_file = model_dir / "label_map.json"
with open(label_map_file, "r") as f:
    tmp = json.load(f)
    id2label = {int(k):v for k, v in tmp["id2label"].items()}
    label2id = tmp["label2id"]
num_types = len(label2id)

SEED = args.seed
EPOCHS = args.epoch
SEQ_LEN = args.gene_num + 1
UNASSIGN = args.novel_type
UNASSIGN_THRES = args.unassign_thres if UNASSIGN == True else 0

BATCH_SIZE = args.batch_size

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        full_seq = data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]

class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

data = sc.read_h5ad(args.data_path)
barcodes = data.obs['barcode'].tolist() if 'barcode' in data.obs else data.obs_names.tolist()
data = data.X
test_dataset = SCDataset(data)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, drop_last=False
)

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = True
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=num_types)

path = model_dir / "finetune_best.pth"
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
model = model.to(device)


model.eval()
pred_proba = np.zeros((data.shape[0], num_types))
with torch.no_grad():
    for i, full_seq in enumerate(test_loader):
        pred_logits = model(full_seq)
        pred_proba[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = pred_logits.cpu().numpy()

from scipy.special import softmax
probs = softmax(pred_proba, axis=-1)
label_names = [id2label[i] for i in range(len(id2label))]
import pandas as pd
df = pd.DataFrame(probs, columns=label_names, index=barcodes)
df.to_csv("scbert_predictions.tsv", sep="\t")
