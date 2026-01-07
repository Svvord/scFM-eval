#!/usr/bin/env python
import argparse
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
sys.path.append(os.path.dirname(__file__))

from datasets import load_from_disk
import torch
from tqdm import tqdm
import numpy as np
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from sccello.src.model_prototype_contrastive import PrototypeContrastiveForSequenceClassification
from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)
from transformers import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='ScCello Fine-tuning Script')
    
    # Model and data parameters
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to the ScCello model')
    parser.add_argument('--tokenized_dataset', type=str, required=True,
                        help='Path to the tokenized dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--eval_size', type=float, default=0.2,
                        help='Evaluation set size (fraction of total data)')
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs.")
    
    return parser.parse_args()


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
    args = parse_args()
    
    # Training arguments
    training_args = TrainingArguments(
        learning_rate=5e-3,
        do_train=True,
        do_eval=True,
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=False,
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=0.001,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=20,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        output_dir="checkpoints",
        label_names=["labels"],
        dataloader_num_workers=16,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit= 2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_from_disk(args.tokenized_dataset)
    labels = np.unique(dataset['cell_type'])
    label2id = {item: i for i, item in enumerate(labels)}
    id2label = {v:k for k, v in label2id.items()}
    def encode(example):
        example["label"] = label2id[example["cell_type"]]
        # 查了一整天bug, 合着它的collator, 逻辑是查询label然后写进batch[labels], 所以config里是labels, 太tm让人迷惑了
        return example
    dataset = dataset.map(encode)
    
    # Train-test split
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=args.eval_size,
        stratify=dataset['cell_type']
    )
    train_dataset = dataset.select(train_idx)
    eval_dataset = dataset.select(test_idx)

    # Load model
    model_kwargs = {
        "num_labels": len(labels),
        "total_logging_steps": training_args.logging_steps * training_args.gradient_accumulation_steps,
        "data_source": "train",
        "normalize_flag": 0,
        "pass_cell_cls": 0,
    }


    model = PrototypeContrastiveForSequenceClassification.from_pretrained(
        args.pretrained_model, **model_kwargs
    ).to(device)

    model.config.label2id = label2id
    model.config.id2label = id2label
    
    # Freeze BERT parameters
    for param in model.bert.parameters():
        param.requires_grad = False


    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCellClassification(
            model_input_name=["input_ids"],
            pad_to_multiple_of=None
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Train the model
    trainer.train()

    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d")
    trainer.save_model(f"./sccello_finetuned_model")


if __name__ == "__main__":
    main()