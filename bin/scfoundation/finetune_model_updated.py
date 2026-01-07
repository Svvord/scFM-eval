import os
import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir) # path to this folder
import numpy as np
import torch
from torch import nn
from load import *
import argparse
import scanpy as sc
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
from dataclasses import dataclass
from pathlib import Path

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    先 linear warmup，再 cosine decay 的 scheduler。
    - warmup_steps: 线性从 0 → base_lr
    - max_steps: 总训练步数
    """
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # 线性 warmup
                lr = base_lr * float(step) / float(max(1, self.warmup_steps))
            else:
                # cosine decay
                progress = float(step - self.warmup_steps) / float(
                    max(1, self.max_steps - self.warmup_steps)
                )
                progress = min(1.0, max(0.0, progress))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine_decay
            lrs.append(lr)
        return lrs


class FineTuneDataset(Dataset):
    def __init__(self, adata):
        if issparse(adata.X):
            self.data = adata.X.toarray()
        else:
            self.data = adata.X
        self.labels = adata.obs['cell_type'].tolist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LinearProbingClassifier(nn.Module):

    def __init__(self, n_class, ckpt_path,frozenmore=True):
        super().__init__()
        self.n_class = n_class
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore

    def build(self):
        model,model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        if self.frozenmore:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        for na, param in self.encoder.transformer_encoder[-2].named_parameters():
            print('self.encoder.transformer_encoder ',na,' have grad')
            param.requires_grad = True


        self.fc1 = nn.Sequential(
        nn.Linear(model_config['encoder']['hidden_dim'], 256),
        nn.ReLU(),
        nn.Linear(256, self.n_class)  # ['n_class']
        ) 
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config
        
    def forward(self, x, *args, **kwargs):
        
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x,x_padding)

        # mlp
        logits, _ = torch.max(logits, dim=1)  # b,dim

        logits = self.norm(logits)
        logits = self.fc1(logits)

        return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to the h5ad file')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to the checkpoint file')
    parser.add_argument('--frozenmore', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=4)

    args = parser.parse_args()

    save_path = Path("./scfoundation_finetuned_model")
    save_path.mkdir()
    save_path = save_path / "best-model.ckpt"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    adata = sc.read_h5ad(args.data_path)
    labels = adata.obs['cell_type'].tolist()
    label2id = {label: i for i, label in enumerate(np.unique(labels))}
    id2label = {i: label for label, i in label2id.items()}
    adata.obs['cell_type'] = adata.obs['cell_type'].map(label2id)
    train_idx, val_idx = train_test_split(
        range(len(labels)), test_size=args.val_size,
        random_state=args.seed, stratify=labels,
    )
    dataset = FineTuneDataset(adata)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2 * args.batch_size , 
        # 很奇怪, 验证的时候 = 1/4 train bs显存都会爆, 似乎模型本身有bug?
        # 把 eval 过程注释掉, 就能训练完. 意味着 eval 比 train 显存多, 特别古怪
        # 没办法, 推理的时候弄成半精度了, 已经几乎试完了所有的优化, 都不work, 可能源码写了training 和 eval过程有什么不一样的. 
        shuffle=True
    )
    
    model = LinearProbingClassifier(len(label2id), args.ckpt_path, args.frozenmore)
    model.build()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(model)
    
    trainable_encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    param_groups = [
        {"params": trainable_encoder_params,"lr": 5e-5,},
        {"params": model.fc1.parameters(),"lr": 5e-5,},
        {"params": model.norm.parameters(),"lr": 5e-5,}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_train_steps = len(train_loader) * args.epochs
    # scheduler = WarmupCosineScheduler(
    #     optimizer,
    #     warmup_steps=args.warmup_steps,
    #     max_steps=total_train_steps,
    #     min_lr=1e-5,
    # )

    criterion = torch.nn.CrossEntropyLoss()

    min_val_loss = float('inf')
    early_stop_patience = args.early_stop_patience
    early_stop_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            print(f'Epoch {epoch}, Train Loss: {loss.item()}')
        del data, labels, loss, outputs
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch in val_loader:
                    data, labels = batch
                    data = data.to(device)
                    labels = labels.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    print(f'Epoch {epoch}, Val Loss: {loss.item()}')

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    early_stop_counter = 0
                    ckpt = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # "scheduler_state_dict": scheduler.state_dict(),
                        "label2id": label2id,
                        "id2label": id2label,
                        "model_config": getattr(model, "model_config", None),
                    }
                    torch.save(ckpt, save_path)
                    print(f">>> New best ckpt saved to: {save_path} (val_loss={min_val_loss:.4f})")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f'Early stopping at epoch {epoch}')
                        # torch.save(model.state_dict(), os.path.join(args.ckpt_path, f'last_model.ckpt'))
                        break

                del data, labels, loss, outputs
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        