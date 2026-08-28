"""MindSpore-CellFM-style cell-type annotation head, reimplemented in PyTorch.

The original MindSpore annotation model (biomed-AI/CellFM
tutorials/CellAnnotation/annotation_model.py) uses learnable per-class prototype
embeddings (`cluster_emb`) that cross-attend over the cell's gene tokens, then
produces two logit pathways:

  labelpred1 = classifier(cross-attended cluster tokens)      # [B, num_cls]
  labelpred2 = cls_token @ cluster_emb.T                       # [B, num_cls]

and trains both with a class-weighted NLL loss. This module ports that head on
top of the existing PyTorch CellFM encoder (`Cell_FM`), reusing the vendored
`CrossRetentionLayer`. Inference uses NO masking / NO downsampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Cell_FM
from layers.torch_retention import CrossRetentionLayer


class AnnotationHead(nn.Module):
    def __init__(self, enc_dims, num_heads, num_cls, n_layers=2, dropout=0.0):
        super().__init__()
        self.num_cls = num_cls
        self.cluster_emb = nn.Parameter(torch.empty(num_cls, enc_dims))
        nn.init.xavier_normal_(self.cluster_emb, gain=0.5)
        self.query_layer = nn.ModuleList([
            CrossRetentionLayer(enc_dims, num_heads, dropout) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(enc_dims, 1, bias=False)

    def forward(self, cls_token, expr_emb, gene_mask=None):
        # cls_token: [B, D]; expr_emb (gene tokens): [B, L, D]; gene_mask: [B, L] valid=1
        b, d = cls_token.shape
        clst = torch.cat([
            cls_token.reshape(b, 1, d),
            self.cluster_emb.unsqueeze(0).expand(b, -1, -1),
        ], dim=1)  # [B, 1 + num_cls, D]
        attn_mask = gene_mask.view(b, 1, -1, 1) if gene_mask is not None else None
        for layer in self.query_layer:
            clst = layer(clst, expr_emb, attn_mask=attn_mask)
        cls_out, cluster = clst[:, 0], clst[:, 1:]
        labelpred1 = self.classifier(cluster).reshape(b, -1)   # [B, num_cls]
        labelpred2 = cls_out @ self.cluster_emb.t()            # [B, num_cls]
        return labelpred1, labelpred2


class Annotation_Cell_FM(nn.Module):
    """CellFM encoder + MindSpore-style annotation head. No reconstruction loss."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_cls = cfg.num_cls
        self.extractor = Cell_FM(27855, cfg, ckpt_path=cfg.ckpt_path, device=cfg.device)
        self.head = AnnotationHead(cfg.enc_dims, cfg.enc_num_heads, cfg.num_cls)

    def forward(self, raw_nzdata=None, dw_nzdata=None, ST_feat=None,
                nonz_gene=None, mask_gene=None, zero_idx=None):
        # raw_nzdata / mask_gene are accepted for call-site compatibility but unused:
        # annotation runs with no masking and no reconstruction loss.
        emb, _ = self.extractor.net.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
        cls_token = emb[:, 0]        # [B, D]
        expr_emb = emb[:, 3:]        # [B, L, D] gene tokens (after cls + 2 ST tokens)
        return self.head(cls_token, expr_emb, gene_mask=zero_idx)

    @staticmethod
    def compute_loss(labelpred1, labelpred2, target, weight=None):
        return (F.cross_entropy(labelpred1, target, weight=weight)
                + F.cross_entropy(labelpred2, target, weight=weight))

    @staticmethod
    def predict_proba(labelpred1, labelpred2):
        # ensemble of both pathways (matches training both with equal weight)
        return 0.5 * (F.softmax(labelpred1, dim=1) + F.softmax(labelpred2, dim=1))
