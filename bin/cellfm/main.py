import scanpy as sc
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from layers.utils import *
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import json

import pickle

import warnings
warnings.filterwarnings("ignore")

from model import Finetune_Cell_FM


def basic(args):
    ### CellFM param ###
    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = args.batch_size
    cfg.mask_ratio = 0.5
    ### Main param ###
    cfg.dataset = args.dataset
    cfg.feature_col = args.feature_col
    cfg.ckpt_path = args.ckpt_path
    cfg.device = args.device
    cfg.epoch = args.epoch 
    cfg.num_cls = 1
    
    #### A lots of Path ####
    PT_PATH = f"../data_pt/{cfg.dataset}"
    MODEL_PATH = f"../model_checkpoint/{cfg.dataset}" # for 40000 cells
    
    #### Make dir ####
    os.makedirs(PT_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    def load_data(adata_path, mode="train"):
        adata = read_h5ad(adata_path)
        # adata.var_names = adata.var['gene_name']
        adata.obs['celltype'] = adata.obs['cell_type']
        adata.obs['feat'] = adata.obs[cfg.feature_col].cat.codes.values
        cfg.num_cls = len(adata.obs['feat'].unique())
        
        adata.obs['batch_id'] = 0
        if mode == "train":
            adata.obs['train'] = 0
            dataset = SCrna(adata, mode="train")
            prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
            loader = build_dataset(
                dataset,
                prep=prep,
                batch_size=cfg.use_bs,
                pad_zero=cfg.pad_zero,
                drop=False,
                shuffle=True
            )
        if mode== "test":
            adata.obs['train'] = 2
            dataset = SCrna(adata, mode="test")
            prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
            loader = build_dataset(
                dataset,
                prep=prep,
                batch_size=cfg.use_bs,
                drop=False,
                shuffle=False
            )
        return loader
    ################### training ###################
    train_adata_path = f"/data/user/liwb/project/CellFM/datasets/cell_annotion/Inter/{cfg.dataset}/train.h5ad"
    test_adata_path = f"/data/user/liwb/project/CellFM/datasets/cell_annotion/Inter/{cfg.dataset}/test.h5ad"
    
    train_loader = load_data(train_adata_path, mode="train")
    test_loader = load_data(test_adata_path, mode="test")
    

    net = Finetune_Cell_FM(cfg) # 27855

    for name, param in net.named_parameters():
        param.requires_grad = "cls." in name or "encoder" in name
    
    print("Trainable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    net = net.to(cfg.device)
    net.extractor.load_model(weight=True, moment=False)
    
    optimizer = AdamW([p for p in net.parameters() if p.requires_grad], 
                      lr=1e-4,
                      weight_decay=1e-5)
    
    scaler = GradScaler() 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    
    criterion_cls = nn.CrossEntropyLoss()
    for epoch in range(cfg.epoch):
        net.train()
        print("training...")
        running_loss = 0.0
        running_acc = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epoch}")
        
        for step, batch in enumerate(progress):    
            
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)
            celltype_label = batch['celltype_label'].to(cfg.device)
            batch_id = batch['batch_id'].to(cfg.device)
            feat = batch['feat'].long().to(cfg.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                cls, mask_loss, cls_token = net(
                    raw_nzdata=raw_nzdata,
                    dw_nzdata=dw_nzdata,
                    ST_feat=ST_feat,
                    nonz_gene=nonz_gene,
                    mask_gene=mask_gene,
                    zero_idx=zero_idx
                ) 
                
                cls_loss = criterion_cls(cls, feat)
                loss = mask_loss + cls_loss
            
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            accuracy = (cls.argmax(1) == feat).sum().item()
            accuracy = accuracy / len(batch_id)
            
            running_loss += loss.item()
            running_acc += accuracy
            
            avg_loss = running_loss / (step + 1)
            avg_acc = running_acc / (step + 1)
            
            progress.set_postfix(loss=avg_loss, acc=avg_acc)
        
        scheduler.step()
        print(f"Epoch {epoch+1} 完成,平均loss: {avg_loss:.6f}")
        torch.save(net.state_dict(), f"{MODEL_PATH}/checkpoint_epoch_{epoch+1}.pth")

        net.eval()
        print("testing...")
        running_loss = 0.0
        running_acc = 0.0
    
        progress = tqdm(test_loader, desc="Testing")
        with torch.no_grad(): 
            for step, batch in enumerate(progress):    
                
                raw_nzdata = batch['raw_nzdata'].to(cfg.device)
                dw_nzdata = batch['dw_nzdata'].to(cfg.device)
                ST_feat = batch['ST_feat'].to(cfg.device)
                nonz_gene = batch['nonz_gene'].to(cfg.device)
                mask_gene = batch['mask_gene'].to(cfg.device)
                zero_idx = batch['zero_idx'].to(cfg.device)
                celltype_label = batch['celltype_label'].to(cfg.device)
                batch_id = batch['batch_id'].to(cfg.device)
                feat = batch['feat'].long().to(cfg.device)

                with torch.cuda.amp.autocast():
                    cls, mask_loss, cls_token = net(
                        raw_nzdata=raw_nzdata,
                        dw_nzdata=dw_nzdata,
                        ST_feat=ST_feat,
                        nonz_gene=nonz_gene,
                        mask_gene=mask_gene,
                        zero_idx=zero_idx
                    ) 
                    
                    cls_loss = criterion_cls(cls, feat)
                    loss = mask_loss[0] + cls_loss

                pred = cls.argmax(1)
                accuracy = (pred == feat).sum().item()
                accuracy = accuracy / len(batch_id)
                
                running_loss += loss.item()
                running_acc += accuracy
                
                avg_loss = running_loss / (step + 1)
                avg_acc = running_acc / (step + 1)
                
                progress.set_postfix(loss=avg_loss, acc=avg_acc)
        
        print(f"Testing {epoch+1} complete,avg loss: {avg_loss:.6f}, avg acc: {avg_acc:.6f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pancrm0")
    parser.add_argument("--feature_col", type=str, default="cell_type")
    parser.add_argument("--ckpt_path", type=str, default="/bigdat2/user/shanggny/checkpoint/para80m/6300w_18000_19479-1_38071.ckpt")
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    basic(args)
