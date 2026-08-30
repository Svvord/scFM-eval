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
from input_normalization import normalize_log1p
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
        # Raw counts -> log1p(1e4-normalised), the representation the backbone was
        # pretrained on and the one get_embedding.py feeds it (see input_normalization.py).
        self.data = normalize_log1p(adata.X)
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
    parser.add_argument('--label_key', type=str, default='cell_type', help='Key of the label column')
    # ---- training-strategy knobs (defaults = the current warmup+cosine, 1e-4 config) ----
    parser.add_argument('--lr', type=float, default=1e-4, help='peak learning rate for all param groups')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['none', 'warmup_cosine'],
                        help="'none' = constant lr (the original strategy); 'warmup_cosine' = warmup->cosine")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup fraction of total steps')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='cosine floor lr')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='max grad norm (0 = disabled)')

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
    labels = adata.obs[args.label_key].tolist()
    label2id = {label: i for i, label in enumerate(np.unique(labels))}
    id2label = {i: label for label, i in label2id.items()}
    adata.obs['cell_type'] = adata.obs[args.label_key].map(label2id)
    # Crash-safe stratified split: every class keeps >=1 training sample; a class too
    # small to spare one for validation (floor(count*val_size)==0) goes entirely to train.
    def _min_train_split(_labels, _vf, _seed=42):
        _labels = np.asarray(_labels); _rng = np.random.RandomState(_seed)
        _tr, _va = [], []
        for _c in np.unique(_labels):
            _ix = np.where(_labels == _c)[0]; _rng.shuffle(_ix)
            _nv = min(int(len(_ix) * _vf), len(_ix) - 1)
            _va.extend(_ix[:_nv].tolist()); _tr.extend(_ix[_nv:].tolist())
        _rng.shuffle(_tr); _rng.shuffle(_va)
        return np.array(_tr, dtype=int), np.array(_va, dtype=int)
    train_idx, val_idx = _min_train_split(labels, args.val_size, args.seed)
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
    # LR + schedule are configurable (see --lr/--scheduler). Backbone stays frozen (only
    # transformer_encoder[-2] + head trainable); the schedule/clip knobs let us A/B the
    # original constant-5e-5 strategy against warmup->cosine variants fairly.
    param_groups = [
        {"params": trainable_encoder_params, "lr": args.lr},
        {"params": model.fc1.parameters(), "lr": args.lr},
        {"params": model.norm.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_train_steps = max(1, len(train_loader) * args.epochs)
    if args.scheduler == 'warmup_cosine':
        warmup_steps = max(1, int(args.warmup_ratio * total_train_steps))
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=warmup_steps,
            max_steps=total_train_steps, min_lr=args.min_lr,
        )
    else:
        scheduler = None  # constant lr = the original strategy
    print(f">>> strategy: lr={args.lr} scheduler={args.scheduler} "
          f"warmup_ratio={args.warmup_ratio} min_lr={args.min_lr} grad_clip={args.grad_clip}")

    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0   # select best-ckpt + early-stop on val ACCURACY (val_loss is
                          # NaN under the required fp16 autocast; val_acc is robust)
    early_stop_patience = args.early_stop_patience
    early_stop_counter = 0
    best_saved = False    # guard: guarantee a usable ckpt gets written

    def _save_ckpt(epoch, tag, metric):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
            "model_config": getattr(model, "model_config", None),
        }
        torch.save(ckpt, save_path)
        print(f">>> {tag} ckpt saved to: {save_path} ({metric})")

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            data, labels = batch
            if data.size(0) == 1:
                # BatchNorm(train) needs >1 sample; skip a size-1 trailing batch (n_train % batch_size == 1)
                continue
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            print(f'Epoch {epoch}, Train Loss: {loss.item()}')
        del data, labels, loss, outputs
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            # autocast stays ON (the fp32 eval forward OOMs on scFoundation's encoder).
            # Under fp16 the loss overflows to NaN, so we select on val_ACCURACY (robust
            # via argmax) instead of val_loss -- see best-ckpt/early-stop logic below.
            with torch.cuda.amp.autocast():
                for batch in val_loader:
                    data, labels = batch
                    data = data.to(device)
                    labels = labels.to(device)
                    outputs = model(data)
                    # compute the loss in fp32: under autocast the fp16 CrossEntropy
                    # overflowed to NaN, which silently broke best-checkpoint selection
                    # AND early-stopping (NaN counts as "no improvement" -> stops at
                    # ~patience epochs while val_acc is still rising).
                    loss = criterion(outputs.float(), labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(dim=-1) == labels).sum().item()
                    val_total += labels.size(0)
                val_acc = val_correct / max(1, val_total)
                print(f'[DIAG] Epoch {epoch}: val_loss_sum={val_loss:.4f} val_acc={val_acc:.4f} (n_val={val_total})')

                # Select on val ACCURACY (higher is better) -- robust to the fp16 NaN loss.
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                    _save_ckpt(epoch, "New best", f"val_acc={best_val_acc:.4f}")
                    best_saved = True
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f'Early stopping at epoch {epoch}')
                        break

                del data, labels, loss, outputs
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # Fallback: if val_loss was never finite (e.g. divergence -> NaN), no best was saved;
    # persist the final model so prediction always has a checkpoint to load.
    if not best_saved:
        print(">>> WARNING: no best ckpt during training; saving final model as fallback.")
        _save_ckpt(args.epochs - 1, "Fallback (final)", f"best_val_acc={best_val_acc:.4f}")
