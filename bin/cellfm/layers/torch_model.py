import numpy as np
from .torch_retention import *
import torch.nn.init as init

class FFN(nn.Module):
    def __init__(self, in_dims, emb_dims, b=256):
        super().__init__()
        self.w1 = nn.Linear(in_dims, b, bias=False)
        self.act1 = nn.LeakyReLU()
        self.w3 = nn.Linear(b, b, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.table = nn.Linear(b, emb_dims, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        b, l, d = x.shape
        v = x.view(-1, d)
        v = self.act1(self.w1(v))
        v = self.w3(v) + v * self.a
        v = self.softmax(v)
        v = self.table(v)
        return v.view(b, l, -1)


class ValueEncoder(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.value_enc = FFN(1, emb_dims)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, emb_dims))

    def forward(self, x):
        if x.dim() == 3:
            unmask, expr = torch.chunk(x, 2, dim=-1)
            unmasked = self.value_enc(expr) * unmask
            masked = self.mask_emb * (1 - unmask)
            expr_emb = masked + unmasked
        else:
            expr = x.unsqueeze(-1)
            unmask = torch.ones_like(expr)
            expr_emb = self.value_enc(expr)
        return expr_emb, unmask

class ValueDecoder_00(nn.Module):
    def __init__(self, emb_dims, dropout=0.1, zero=False):
        super().__init__()
        self.zero = zero

        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dims),
            nn.Linear(emb_dims, emb_dims),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dims, 1)
        )

        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.LayerNorm(emb_dims),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid()
            )

    def forward(self, expr_emb):
        b, l, d = expr_emb.shape
        pred = self.mlp(expr_emb).view(b, l)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).view(b, l)
        return pred, zero_prob


class ValueDecoder_1(nn.Module):
    def __init__(self, emb_dims, dropout, zero=False):
        super().__init__()
        self.zero = zero

        self.norm = nn.LayerNorm(emb_dims)  # 加入 LayerNorm
        self.w1 = nn.Linear(emb_dims, emb_dims, bias=True)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(emb_dims, 1, bias=True)
        init.xavier_normal_(self.w1.weight, gain=1.0)
        init.xavier_normal_(self.w2.weight, gain=1.0)
        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid(),
            )

    def forward(self, expr_emb):
        b, l, d = expr_emb.shape
        # pred = self.w2(self.act(self.w1(expr_emb))).view(b, l)
        x = self.norm(expr_emb)               # LayerNorm 
        x = self.w1(x)                        # Linear
        x = self.act(x)                       # ReLU 
        x = self.dropout(x)
        pred = self.w2(x).view(b, l)          # output

        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).view(b, l)
        return pred, zero_prob

class CellwiseDecoder_00(nn.Module):
    def __init__(self, in_dims, emb_dims=None, dropout=0.1, zero=False):
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.zero = zero

        self.query_proj = nn.Sequential(
            nn.LayerNorm(in_dims),
            nn.Linear(in_dims, emb_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dims, emb_dims),
        )
        self.key_proj = nn.Linear(in_dims, emb_dims)

        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.LayerNorm(emb_dims),
                nn.Linear(emb_dims, emb_dims),
                nn.ReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.Sigmoid()
            )

    def forward(self, cell_emb, gene_emb):
        b, l, d = gene_emb.shape
        # Project gene embeddings into query space
        query = self.query_proj(gene_emb)  # [B, L, D]
        # Project cell embedding into key space
        key = self.key_proj(cell_emb).view(b, -1, 1)  # [B, D, 1]
        # Inner product as similarity
        pred = torch.bmm(query, key).view(b, l)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(query).bmm(key).view(b, l)
        return pred, zero_prob


class CellwiseDecoder_1(nn.Module):
    def __init__(self, in_dims, emb_dims=None, dropout=0.0, zero=False):
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.zero = zero

        self.norm = nn.LayerNorm(in_dims)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.map = nn.Linear(in_dims, emb_dims)
        init.xavier_normal_(self.map.weight, gain=1.0)
        if zero:
            self.zero_logit = nn.Linear(emb_dims, emb_dims)

    def forward(self, cell_emb, gene_emb):
        b = cell_emb.size(0)
        # query = torch.sigmoid(self.map(gene_emb))
        # query = torch.sigmoid(self.map(self.act(self.norm(gene_emb))))
        # LayerNorm → ReLU → Dropout → Linear → Sigmoid
        x = self.norm(gene_emb)
        x = self.act(x)
        x = self.dropout(x)
        query = torch.sigmoid(self.map(x))  # shape: [b, l, d]
        key = cell_emb.view(b, -1, 1)
        pred = torch.bmm(query, key).view(b, -1)
        if not self.zero:
            return pred
        zero_query = self.zero_logit(gene_emb)
        zero_prob = torch.sigmoid(torch.bmm(zero_query, key)).view(b, -1)
        return pred, zero_prob


class ValueDecoder(nn.Module):
    def __init__(self, emb_dims, dropout, zero=False):
        super().__init__()
        self.zero = zero
        self.w1 = nn.Linear(emb_dims, emb_dims, bias=False)
        self.act = nn.LeakyReLU()
        self.w2 = nn.Linear(emb_dims, 1, bias=False)
        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid(),
            )

    def forward(self, expr_emb):
        b, l, d = expr_emb.shape
        pred = self.w2(self.act(self.w1(expr_emb))).view(b, l)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).view(b, l)
        return pred, zero_prob


class CellwiseDecoder(nn.Module):
    def __init__(self, in_dims, emb_dims=None, dropout=0.0, zero=False):
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.zero = zero
        self.map = nn.Linear(in_dims, emb_dims, bias=False)
        if zero:
            self.zero_logit = nn.Linear(emb_dims, emb_dims)

    def forward(self, cell_emb, gene_emb):
        b = cell_emb.size(0)
        query = torch.sigmoid(self.map(gene_emb))
        key = cell_emb.view(b, -1, 1)
        pred = torch.bmm(query, key).squeeze(-1)
        if not self.zero:
            return pred
        zero_query = self.zero_logit(gene_emb)
        zero_prob = torch.sigmoid(torch.bmm(zero_query, key)).squeeze(-1)
        return pred, zero_prob
