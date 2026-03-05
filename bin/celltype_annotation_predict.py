import torch.nn as nn
import torch
import scanpy as sc
import argparse
from transformers import Trainer, TrainingArguments
from scipy.sparse import issparse
from torch.utils.data import Dataset
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
from celltype_annotation_finetune import CellClassificationModel
from scipy.special import softmax


class EmbeddingDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_embeddings": self.X[idx],
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--method_name', type=str, required=True, help='Method name')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.data_path)
    barcode = adata.obs['barcode'].tolist()
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    dataset = EmbeddingDataset(X)

    model = CellClassificationModel.from_pretrained(args.model_dir)

    training_args = TrainingArguments(
        output_dir="test_output",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        prediction_loss_only=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    pred = trainer.predict(dataset)
    probs = softmax(pred.predictions, axis=-1)

    label_names = [model.config.id2label[str(i)] for i in range(len(model.config.id2label))]
    import pandas as pd
    df = pd.DataFrame(probs, columns=label_names, index=barcode)
    df.to_csv(f"{args.method_name}_predictions.tsv", sep="\t")