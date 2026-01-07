import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)



class SRMSNorm(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.scale = 1.0 / math.sqrt(emb_dims)
        self.eps = 1e-7

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        norm = torch.norm(x * self.scale, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)
        return (x / norm).to(x_dtype)


class DropPath(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if not self.training or self.dropout.p == 0.0:
            return x
        B, L, D = x.shape
        mask = torch.ones(B, 1, 1, device=x.device)
        mask = self.dropout(mask)
        return x * mask


class Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class LoraBlock(nn.Module):
    def __init__(self, in_dim, out_dim, r):
        super().__init__()
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(self.A(x))

    def update_weight(self):
        return self.B.weight @ self.A.weight


class MHRetention(nn.Module):
    def __init__(self, emb_dims, num_heads, lth=None, lora=0):
        super().__init__()
        self.emb_dims = emb_dims
        self.num_heads = num_heads
        self.head_dim = emb_dims // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.lora = lora
        beta = 1.0 if lth is None else (lth * 8) ** -0.25

        self.q_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.k_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)

        self.kernelQ = Kernel()
        self.kernelK = Kernel()
        self.kernelV = nn.Identity()
        self.kernelU = SiLU()

        self.pre_norm = SRMSNorm(emb_dims)
        self.inner_norm = SRMSNorm(self.head_dim)

        if self.lora > 0:
            self.lora_q = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_k = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(self, x, y=None, v_pos=None, attn_mask=None, seq_mask=None):
        if y is None:
            y = x
        B, L1, D = x.shape
        L2 = y.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        u = self.u_proj(x)

        if self.lora > 0:
            q += self.lora_q(x)
            k += self.lora_k(y)
            v += self.lora_v(y)
            u += self.lora_u(x)

        def reshape(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = reshape(q)
        K = reshape(k)
        V = reshape(v)
        U = reshape(u)

        Q = self.kernelQ(Q) / self.scale
        K = self.kernelK(K) / self.scale
        V = self.kernelV(V)
        U = self.kernelU(U)

        if seq_mask is not None:
            Q = Q * seq_mask
        if attn_mask is not None:
            K = K * attn_mask
        if v_pos is not None:
            V = V * v_pos

        KV = torch.matmul(K.transpose(-1, -2), V)
        O = torch.matmul(Q, KV)
        O = self.inner_norm(O) * U

        O = O.transpose(1, 2).contiguous().view(B, L1, D)
        O = self.o_proj(O)
        if self.lora > 0:
            O += self.lora_o(O)
        return O


class GatedLinearUnit(nn.Module):
    def __init__(self, emb_dims, lth=None, lora=0):
        super().__init__()
        beta = 1.0 if lth is None else (lth * 8) ** -0.25
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.norm = SRMSNorm(emb_dims)
        self.lora = lora
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)
        if self.lora > 0:
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        u = self.u_proj(x_flat)
        v = self.v_proj(x_flat)

        if self.lora > 0:
            u += self.lora_u(x_flat)
            v += self.lora_v(x_flat)

        out = u * v
        out = self.o_proj(out)

        if self.lora > 0:
            out += self.lora_o(out)

        # out = self.norm(out)
        return out.view(B, L, D)

class RetentionLayer(nn.Module):
    def __init__(self, emb_dims, num_heads, lth, dropout=0.0, lora=0, recompute=False):
        super().__init__()
        self.alpha = (2 * lth) ** 0.25
        self.attn = MHRetention(emb_dims, num_heads, lth, lora)
        self.ffn = GatedLinearUnit(emb_dims, lth, lora)
        self.dropout = nn.Dropout(p=dropout)
        self.post_norm1 = nn.LayerNorm(emb_dims)
        self.post_norm2 = nn.LayerNorm(emb_dims)

    def forward(self, x, **kwargs):
        out = self.dropout(self.attn(x, **kwargs))
        x = self.post_norm1(x * self.alpha + out)
        out = self.dropout(self.ffn(x))
        x = self.post_norm2(x * self.alpha + out)
        return x


class CrossRetentionLayer(nn.Module):
    def __init__(self, emb_dims, num_heads, dropout=0.0, recompute=False):
        super().__init__()
        self.attn1 = MHRetention(emb_dims, num_heads)
        self.attn2 = MHRetention(emb_dims, num_heads)
        self.ffn = GatedLinearUnit(emb_dims)
        self.dropout = nn.Dropout(p=dropout)
        self.post_norm1 = nn.LayerNorm(emb_dims)
        self.post_norm2 = nn.LayerNorm(emb_dims)
        self.post_norm3 = nn.LayerNorm(emb_dims)

    def forward(self, x, y, v_pos=None, attn_mask=None, seq_mask=None):
        out = self.attn1(x)
        x = self.post_norm1(x + self.dropout(out))
        out = self.attn2(x, y=y, v_pos=v_pos, attn_mask=attn_mask, seq_mask=seq_mask)
        x = self.post_norm2(x + self.dropout(out))
        out = self.ffn(x)
        x = self.post_norm3(x + self.dropout(out))
        return x