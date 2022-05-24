from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn

from cfg import get_cfg
from datasets import get_ds
from methods import get_method

import yaml
import sys
import os
sys.path.insert(0, '../Learnable_invariance_package')
from learnable_invariance import learnable_invariance

def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epoch - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None


if __name__ == "__main__":
    cfg = get_cfg()
    
    #[Li] Load Learnable invariance module
    if cfg.Li_config_path:
        import yaml
        Li_configs=yaml.safe_load(open(cfg.Li_config_path,'r'))
        if cfg.entropy_weights:
            Li_configs['entropy_weights']=cfg.entropy_weights
    else:
        Li_configs={'li_flag': False}
    
    if Li_configs['li_flag']:
        Li=learnable_invariance(Li_configs)
        if Li_configs['load_li']:
            Li.augmentation.get_param.conv.load_state_dict(torch.load(cfg.fname))
        Li.augmentation.get_param.conv.cuda()
    else:
        Li=None
    
    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers) #[Li] Remember to turn off random cropping when using Li
    model = get_method(cfg.method)(cfg)
    model.cuda().train()
    if cfg.fname is not None:
        model.load_state_dict(torch.load(cfg.fname))

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    
    if Li_configs['li_flag']:
        #optimizer_Li=optim.Adam(Li.parameters(), lr=Li_configs['lr']) #[Li]
        optimizer_Li=optim.SGD(Li.parameters(), lr=Li_configs['lr'])
    
    scheduler = get_scheduler(optimizer, cfg)

    eval_every = cfg.eval_every
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True

    acc_knn, acc = model.get_acc(ds.clf, ds.test, Li_configs=Li_configs, Li=Li)
    print({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn})