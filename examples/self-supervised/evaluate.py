#!tem changes in Line 123

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
        Li.augmentation.get_param.conv.cuda()
    else:
        Li=None
        
    if cfg.contrastive_train_aug:
        Li_configs['contrastive_train_aug']=True
    if cfg.contrastive_train_copies>0:
        Li_configs['contrastive_train_copies']=cfg.contrastive_train_copies
    if cfg.contrastive_test_aug:
        Li_configs['contrastive_test_aug']=True
    if cfg.contrastive_test_copies>0:
        Li_configs['contrastive_test_copies']=cfg.contrastive_test_copies
    
    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers) #[Li] Remember to turn off random cropping when using Li
    model = get_method(cfg.method)(cfg)
    if cfg.fname is not None:
        model.load_state_dict(torch.load(cfg.fname))
    if Li is not None and cfg.Li_fname is not None:
        Li.augmentation.get_param.conv.load_state_dict(torch.load(cfg.Li_fname))

   
    
    #samples_memory=None#
    acc_knn, acc = model.get_acc(ds.clf, ds.test, Li_configs=Li_configs, Li=Li)
    print('top-1 accuracy', acc)
