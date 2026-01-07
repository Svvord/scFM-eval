# PyTorch version of FinetuneModel, faithfully mirroring the MindSpore structure
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# You will need to reimplement the following components in PyTorch:
# - ValueEncoder
# - ValueDecoder
# - CellwiseDecoder
# - RetentionLayer
# - SRMSNorm
# - FFN
# These should exactly replicate your MindSpore definitions
from .torch_model import ValueEncoder, ValueDecoder, CellwiseDecoder, FFN
from .torch_retention import RetentionLayer, SRMSNorm
class MaskedMSE(nn.Module):
    def __init__(self, tag=None):
        super().__init__()
        self.tag = tag or ''

    def forward(self, pred, target, mask=None):
        pred = pred.float()
        target = target.float()
        loss = (pred - target) ** 2
        if mask is not None:
            mask = mask.float()
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss
class BCELoss(nn.Module):
    def __init__(self, tag=''):
        super().__init__()
        self.tag = tag
        self.eps = 1e-12

    def forward(self, pred, target, mask=None):
        pred = pred.float().reshape(-1, 1)
        target = target.float().reshape(-1, 1)
        
        pred_cat = torch.cat([1 - pred, pred], dim=-1)
        target_cat = torch.cat([1 - target, target], dim=-1)
        log_pred = torch.log(torch.clamp(pred_cat, min=self.eps, max=1.0))
        logit = -torch.sum(log_pred * target_cat, dim=-1)
        
        # pred = torch.log(torch.clamp(pred, min=self.eps, max=1.0))
        # target_cat = torch.cat([1 - target, target], dim=-1)
        # logit = -torch.sum(pred * target_cat, dim=-1)

        if mask is not None:
            mask = mask.float().reshape(-1)
            loss = (logit * mask).sum() / mask.sum()
        else:
            loss = logit.mean()

        return loss


class FinetuneModel(nn.Module):
    def __init__(self, n_genes, cfg):
        super().__init__()
        self.depth = cfg.enc_nlayers
        self.if_cls = False
        self.n_genes = n_genes
        self.add_zero = cfg.add_zero and not cfg.pad_zero
        self.pad_zero = cfg.pad_zero
        self.ecs = cfg.ecs
        self.ecs_threshold = cfg.ecs_threshold

        self.gene_emb = nn.Parameter(torch.empty(n_genes + 1 + (-n_genes - 1) % 8, cfg.enc_dims))
        self.ST_emb = nn.Parameter(torch.empty(1, 2, cfg.enc_dims))
        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.enc_dims))
        self.zero_emb = nn.Parameter(torch.zeros(1, 1, cfg.enc_dims))
        nn.init.xavier_normal_(self.gene_emb)
        nn.init.xavier_normal_(self.ST_emb)
        nn.init.xavier_normal_(self.cls_token)
        with torch.no_grad():
            self.gene_emb[0, :] = 0

        self.value_enc = ValueEncoder(cfg.enc_dims)
        self.ST_enc = FFN(1, cfg.enc_dims)
        self.encoder = nn.ModuleList([
            RetentionLayer(cfg.enc_dims, cfg.enc_num_heads, cfg.enc_nlayers,
                           cfg.enc_dropout * i / cfg.enc_nlayers, cfg.lora, cfg.recompute)
            for i in range(cfg.enc_nlayers)
        ])
        self.value_dec = ValueDecoder(cfg.enc_dims, dropout=cfg.dropout, zero=self.add_zero)
        self.cellwise_dec = CellwiseDecoder(cfg.enc_dims, cfg.enc_dims, dropout=cfg.dropout, zero=self.add_zero)

        self.reconstruct1 = MaskedMSE(tag='_ge')
        self.reconstruct2 = MaskedMSE(tag='_ce')
        self.bce_loss1 = BCELoss(tag='_ge')
        self.bce_loss2 = BCELoss(tag='_ce')

    @torch.no_grad()
    def embedding_infer(self, expr, gene, ST_feat, zero_idx):
        b, l = gene.shape
        gene_emb = self.gene_emb[gene] 
        expr_emb, unmask = self.value_enc(expr)
        len_scale = torch.rsqrt(zero_idx.sum(dim=-1, keepdim=True).float() - 3 + 1e-6)
        len_scale = len_scale.view(b, 1, 1, 1).detach()
        if not self.pad_zero:
            zero_unmask = (1 - zero_idx).unsqueeze(-1) * unmask
            expr_emb = zero_unmask * self.zero_emb + (1 - zero_unmask) * expr_emb
        expr_emb = gene_emb + expr_emb

        if ST_feat is None:
            cls_token = self.cls_token.expand(b, -1, -1)
            expr_emb = torch.cat([cls_token, expr_emb], dim=1)
            zero_idx = torch.cat([torch.ones((b, 1), device=zero_idx.device), zero_idx], dim=1)
            if self.pad_zero:
                expr_emb = expr_emb * zero_idx.unsqueeze(-1)
            mask_pos = torch.cat([torch.ones((b, 1, 1), device=unmask.device), unmask], dim=1).unsqueeze(1)
            for i in range(self.depth // 2):
                expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)
            mask_pos = zero_idx.view(zero_idx.size(0), 1, -1, 1) if self.pad_zero else None
            for i in range(self.depth // 2, self.depth):
                expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)
            return expr_emb, gene_emb
        else:
            raise Exception("ST ERROR...")

    @torch.no_grad()
    def decode_infer(self, cls_token, gene_emb, expr_emb):
        # 两路解码器
        if self.add_zero:
            gw_pred, _ = self.value_dec(expr_emb)                   # [B, L, 1]
            cw_pred, _ = self.cellwise_dec(cls_token, gene_emb)     # [B, L, 1]
            return gw_pred, cw_pred
        gw_pred = self.value_dec(expr_emb)  # [B, L, 1]
        cw_pred = self.cellwise_dec(cls_token, gene_emb)  # [B, L, 1]
        return gw_pred, cw_pred

    @torch.no_grad()
    def inference(self, raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx, base_mask=None):
        if ST_feat is None:
            emb, gene_emb = self.embedding_infer(dw_nzdata, nonz_gene, ST_feat, zero_idx)
            cls_token,  expr_emb = emb[:, 0],  emb[:, 1:]
        else:
            emb, gene_emb = self.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
            cls_token, _, expr_emb = emb[:, 0], emb[:, 1:3], emb[:, 3:]
        gw_pred, cw_pred = self.decode_infer(cls_token, gene_emb, expr_emb)
        loss1 = self.reconstruct1(gw_pred, raw_nzdata, mask_gene)
        # loss2 = self.reconstruct2(cw_pred, raw_nzdata, mask_gene)
        total_loss = self.reconstruct1(gw_pred, raw_nzdata, None)
        gene_loss = self.reconstruct1(gw_pred, raw_nzdata, base_mask)
        nonz_gene_loss = self.reconstruct1(gw_pred, raw_nzdata, zero_idx)
        # loss = loss1 + loss2
        return expr_emb, gw_pred, cw_pred, loss1, nonz_gene_loss, gene_loss, total_loss


    def encode(self, expr, gene, ST_feat, zero_idx):
        b, l = gene.shape
        gene_emb = self.gene_emb[gene] 
        expr_emb, unmask = self.value_enc(expr)
        # len_scale = torch.rsqrt(zero_idx.sum(-1).float() - 3 + 1e-6).view(b, 1, 1, 1).detach()
        len_scale = torch.rsqrt(zero_idx.sum(dim=-1, keepdim=True).float() - 3 + 1e-6)
        len_scale = len_scale.view(b, 1, 1, 1).detach()
        if not self.pad_zero:
            zero_unmask = (1 - zero_idx).unsqueeze(-1) * unmask
            expr_emb = zero_unmask * self.zero_emb + (1 - zero_unmask) * expr_emb

        expr_emb = gene_emb + expr_emb
        if self.training:
            st_emb = self.ST_enc(ST_feat.reshape(b, -1, 1))
            st_emb = self.ST_emb + st_emb
        else:
            st_emb = self.ST_emb
            st_emb = st_emb.expand(b, -1, -1)
        cls_token = self.cls_token.expand(b, -1, -1)
        expr_emb = torch.cat([cls_token, st_emb, expr_emb], dim=1)
        
        zero_idx = torch.cat([torch.ones((b, 3), device=zero_idx.device), zero_idx], dim=1)

        if self.pad_zero:
            expr_emb = expr_emb * zero_idx.unsqueeze(-1)

        mask_pos = torch.cat([torch.ones((b, 3, 1), device=unmask.device), unmask], dim=1).unsqueeze(1)
        for i in range(self.depth // 2):
            expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)
        
        mask_pos = zero_idx.view(zero_idx.size(0), 1, -1, 1) if self.pad_zero else None
        for i in range(self.depth // 2, self.depth):
            expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)
        # simple deal with nan
        expr_emb = torch.where(
            torch.isnan(expr_emb),
            torch.nanmean(expr_emb, dim=0),
            expr_emb
        )
        return expr_emb, gene_emb

    def forward(self, raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx, *args):
        emb, gene_emb = self.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
        
        cls_token, st_emb, expr_emb = emb[:, 0], emb[:, 1:3], emb[:, 3:]
        # expr_emb, gene_emb, cls_token = self.embedding_forward(dw_nzdata, nonz_gene, ST_feat, zero_idx)

        if self.add_zero:
            gw_pred, z_prob1 = self.value_dec(expr_emb)
            cw_pred, z_prob2 = self.cellwise_dec(cls_token, gene_emb)
        else:
            gw_pred = self.value_dec(expr_emb)
            cw_pred = self.cellwise_dec(cls_token, gene_emb)

        # use_mask = dw_nzdata[:, :, 0]
        # print(use_mask.sum(dim=1), mask_gene.sum(dim=1))
        # loss1 = self.reconstruct1(gw_pred, raw_nzdata, mask_gene*zero_idx)
        # loss2 = self.reconstruct2(cw_pred, raw_nzdata, mask_gene*zero_idx)

        mask = mask_gene
        loss1 = self.reconstruct1(gw_pred, raw_nzdata, mask)
        loss2 = self.reconstruct2(cw_pred, raw_nzdata, mask)
        loss = loss1 + loss2
        if self.training and self.add_zero:
            nonz_pos = zero_idx
            loss3 = self.bce_loss1(z_prob1, nonz_pos, mask_gene)
            loss4 = self.bce_loss2(z_prob2, nonz_pos, mask_gene)
            loss = loss + loss3 + loss4

        if self.training and self.ecs:
            # cell_emb_normed = self.norm1(cw_pred, p=2, dim=1)
            cell_emb_normed = F.normalize(cw_pred, p=2, dim=1)
            cos_sim = torch.matmul(cell_emb_normed, cell_emb_normed.transpose(0, 1))
            eye_mask = torch.eye(cos_sim.size(0), device=cos_sim.device, dtype=cos_sim.dtype)
            cos_sim = cos_sim * (1 - eye_mask)
            cos_sim = F.relu(cos_sim)
            loss += torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        return loss if self.training else (loss1, loss2)


if __name__ == '__main__':
    class Config_80M:
        def __init__(self):
            self.start_lr = 1e-7
            self.max_lr = 1e-6
            self.min_lr = 5e-7
            self.factor = 5
            self.lora = 0
            self.alpha = 0
            self.lamb = 10
            self.nb_features = 256
            self.nonz_len = 2048
            self.mask_len = 2048
            self.filt_len = 200
            self.dropout = 0.1

            self.enc_dims = 1536
            self.enc_nlayers = 2
            self.enc_num_heads = 48
            self.enc_dropout = 0.1

            self.dec_dims = 512
            self.dec_nlayers = 6
            self.dec_num_heads = 16
            self.dec_dropout = 0.1

            self.temp = 0.2
            self.eps = 1e-2
            self.recompute = True
            self.sim = 0.8
            self.add_zero = True
            self.pad_zero = False
            self.label = False

            self.num_cls = 80
            self.platforms = 27
            self.ttl_step = 1e5

    cfg=Config_80M()
    cfg.ecs_threshold=0.8
    cfg.add_zero=True
    cfg.pad_zero=True
    cfg.ecs=True
    cfg.enc_nlayers=2

    from mindspore.train.serialization import load_checkpoint
    net = FinetuneModel(27855, cfg)
    def map_ms_to_pt(ms_key):
        name = ms_key
        name = name.replace("layer_norm.gamma", "weight")
        name = name.replace("layer_norm.beta", "bias")
        name = name.replace("post_norm1.gamma", "post_norm1.weight")
        name = name.replace("post_norm1.beta", "post_norm1.bias")
        name = name.replace("post_norm2.gamma", "post_norm2.weight")
        name = name.replace("post_norm2.beta", "post_norm2.bias")
        return name


    ms_ckpt = load_checkpoint("CellFM_80MB_MS_Finetune-257_100.ckpt")
    # ms_ckpt = load_checkpoint("../CellFM_80M_weight.ckpt")
    torch_state_dict = {}
    moment_state_dict_1 = {}
    moment_state_dict_2 = {}

    for ms_key, ms_param in ms_ckpt.items():
        pt_key = map_ms_to_pt(ms_key)
        pt_tensor = torch.tensor(ms_param.asnumpy())

        if pt_key.startswith("moment1."):
            param_name = pt_key[len("moment1."):]
            moment_state_dict_1[param_name] = pt_tensor
        elif pt_key.startswith("moment2."):
            param_name = pt_key[len("moment2."):]
            moment_state_dict_2[param_name] = pt_tensor
        elif pt_key in ['global_step', 'learning_rate', 'beta1_power', 'beta2_power',
                        'current_iterator_step', 'last_overflow_iterator_step']:
            continue  # 忽略这些非参数项
        else:
            torch_state_dict[pt_key] = pt_tensor

    missing_keys, unexpected_keys = net.load_state_dict(torch_state_dict, strict=False)
    print(f"[Load Report]")
    print(f"Missing keys in PyTorch model: {missing_keys}")
    print(f"Unexpected keys in checkpoint: {unexpected_keys}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0,  
    )

    # =========================================================================
    # Build param_name → param tensor mapping
    param_name_map = {name: param for name, param in net.named_parameters()}
    # Build PyTorch optimizer state (note: param is used as the key)
    moment_states = {}
    for name, param in param_name_map.items():
        if name in moment_state_dict_1 and name in moment_state_dict_2:
            # Force device and dtype to match param
            exp_avg = moment_state_dict_1[name].clone().to(param.device).to(param.dtype)
            exp_avg_sq = moment_state_dict_2[name].clone().to(param.device).to(param.dtype)
            step = torch.tensor(0, dtype=torch.float32, device='cpu')  # Recommended to keep on CPU
            moment_states[param] = {
                'exp_avg': exp_avg,
                'exp_avg_sq': exp_avg_sq,
                'step': step
            }
    optimizer.state = moment_states
    # =========================================================================
    meta_info = {}
    for key in ['global_step', 'learning_rate', 'beta1_power', 'beta2_power',
                'current_iterator_step', 'last_overflow_iterator_step']:
        if key in ms_ckpt:
            val = ms_ckpt[key].asnumpy()
            if val.shape == ():  # Scalar
                val = val.item()
            meta_info[key] = val
    def restore_meta_to_torch(meta_info, optimizer):
        step_tensor = torch.tensor(meta_info['global_step'], dtype=torch.float32, device='cpu')
        # Restore learning rate
        for group in optimizer.param_groups:
            group['lr'] = meta_info.get('learning_rate', group['lr'])
        # Restore step
        for param in optimizer.state:
            optimizer.state[param]['step'] = step_tensor

    # restore_meta_to_torch(meta_info, optimizer)
    # =========================================================================
    # print(net)

