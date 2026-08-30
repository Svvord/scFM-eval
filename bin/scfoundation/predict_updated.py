import os
import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir) # path to this folder
import numpy as np
import torch
from torch import nn
from finetune_model_updated import LinearProbingClassifier
import argparse
import scanpy as sc
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from scipy.sparse import issparse
from input_normalization import normalize_log1p
from pathlib import Path

class PredictDataset(Dataset):
    def __init__(self, adata):
        # Raw counts -> log1p(1e4-normalised), the representation the backbone was
        # pretrained on and the one get_embedding.py feeds it (see input_normalization.py).
        self.data = normalize_log1p(adata.X)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--backbone_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        os.path.join(args.ckpt_path, "best-model.ckpt"), 
        map_location="cpu", 
        weights_only=False
    )
    # 恢复 label 映射
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]

    # 重新构建模型（注意 ckpt["args"] 里也有当时的参数）
    model = LinearProbingClassifier(
        n_class=len(label2id),
        ckpt_path=args.backbone_path,
        frozenmore=True,
    )
    model.build()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    print(model)
    
    adata = sc.read_h5ad(args.data_path)
    dataset = PredictDataset(adata)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logits = np.zeros((len(dataset), len(label2id)))
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i, batch in enumerate(dataloader):
                batch = batch.to(device)
                outputs = model(batch)
                logits[i * args.batch_size:(i+1) * args.batch_size] = outputs.cpu().numpy()
        
    from scipy.special import softmax
    pred_proba = softmax(logits, axis=-1)
    label_names = [id2label[i] for i in range(len(label2id))]
    barcodes = adata.obs_names.tolist()
    import pandas as pd
    df = pd.DataFrame(pred_proba, index=barcodes, columns=label_names)
    df.to_csv("scfoundation_predictions.tsv", sep="\t")
