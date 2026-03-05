from transformers import PretrainedConfig
from transformers import PreTrainedModel
import torch.nn as nn
import torch
import scanpy as sc
import argparse
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_embeddings": self.X[idx],
            "labels": self.y[idx]
        }

class CellClassificationConfig(PretrainedConfig):
    model_type = "cell_classifier"

    def __init__(
        self,
        in_dim: int=1024,
        num_labels: int=10,
        id2label=None,
        label2id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.num_labels = num_labels

        if id2label is None and label2id is None:
            id2label = {i: str(i) for i in range(num_labels)}
            label2id = {v: k for k, v in id2label.items()}
        elif id2label is None:
            id2label = {i: l for l, i in label2id.items()}
        elif label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        self.id2label = id2label
        self.label2id = label2id


class CellClassificationModel(PreTrainedModel):
    config_class = CellClassificationConfig

    def __init__(self, config: CellClassificationConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        in_dim = config.in_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, config.num_labels),
        )

        self.post_init()

    def forward(
        self,
        input_embeddings=None,
        labels=None,
        **kwargs,
    ):
        logits = self.classifier(input_embeddings)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
        

def compute_metrics(eval_preds):
    probs, labels = eval_preds
    preds = np.argmax(probs, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    try:
        if probs.shape[1] > 2:
            auroc = roc_auc_score(labels, probs, multi_class="ovo", average='macro') # insensitive to class imbalance
        else:
            auroc = roc_auc_score(labels, probs[:, 1], average='macro') # insensitive to class imbalance
    except:
        auroc = 0

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'auroc': auroc
    }

def preprocess_logits_for_metrics(logits, labels):
    probs = torch.softmax(logits, dim=-1)
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--method_name', type=str, required=True, help="Method name")
    parser.add_argument('--label_key', type=str, default='cell_type', help='Key of the label column')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=50000, help='Maximum number of training steps')
    parser.add_argument('--eval_size', type=float, default=0.2)
    parser.add_argument('--es_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ============= load data ============= #
    adata = sc.read_h5ad(args.data_path)
    if args.label_key not in adata.obs.columns:
        raise ValueError(f"Label key {args.label_key} not found in adata.obs")
    cell_types = adata.obs[args.label_key].unique()
    label2id = {cell_type: i for i, cell_type in enumerate(cell_types)}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(cell_types)
    in_dim = adata.X.shape[1]

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    y = np.array([label2id[ct] for ct in adata.obs[args.label_key].tolist()])

    train_idx, eval_idx = train_test_split(
        range(len(adata)),
        test_size=args.eval_size,
        stratify=adata.obs[args.label_key].tolist(),
        random_state=args.seed
    )

    train_dataset = EmbeddingDataset(X[train_idx], y[train_idx])
    eval_dataset = EmbeddingDataset(X[eval_idx], y[eval_idx])

    # ============= load model ============= #
    config = CellClassificationConfig(
        in_dim=in_dim, num_labels=num_labels,
        label2id=label2id, id2label=id2label
    )
    model = CellClassificationModel(config)

    # ============= train model ============= #
    warmup_steps = int(args.max_steps * 0.005)

    training_args = TrainingArguments(
        learning_rate=1e-3,
        weight_decay=1e-4,
        do_train=True,
        do_eval=True,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_steps=args.max_steps,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        output_dir='./checkpoints',
        dataloader_num_workers=4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4 * args.batch_size,
        save_total_limit=2,
        report_to="none",
        # 实际测试下来, 还是 loss 比较好
        # metric_for_best_model="auroc",
        # greater_is_better=True,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.es_patience
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping_callback]
    )
    trainer.train()
    trainer.save_model(f'{args.method_name}_finetuned_model')

if __name__ == "__main__":
    main()