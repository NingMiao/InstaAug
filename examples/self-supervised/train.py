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

import sys
sys.path.insert(0, '../../')
from InstaAug_module import learnable_invariance

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
    
    #Mkdir to save checkpoints
    if not os.path.exists(cfg.model_folder.split('/')[0]):
        os.mkdir(cfg.model_folder.split('/')[0])
    if not os.path.exists(cfg.model_folder):
        os.mkdir(cfg.model_folder)
    
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
    
    wandb.init(project=cfg.wandb, config=cfg, name=cfg.wandb_name)

    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers) #[Li] Remember to turn off random cropping when using Li
    model = get_method(cfg.method)(cfg)
    model.cuda().train()
    if cfg.fname is not None:
        model.load_state_dict(torch.load(cfg.fname))
    if Li is not None and cfg.Li_fname is not None:
        Li.augmentation.get_param.conv.load_state_dict(torch.load(cfg.Li_fname))

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    
    if Li_configs['li_flag']:
        #optimizer_Li=optim.Adam(Li.parameters(), lr=Li_configs['lr']) #[Li]
        optimizer_Li=optim.SGD(Li.parameters(), lr=Li_configs['lr'])
    
    scheduler = get_scheduler(optimizer, cfg)

    eval_every = cfg.eval_every
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True
    start_entropy=None#!#!#!
    
    if 'no_train_period' in Li_configs and Li_configs['no_train_period']>0:
        no_train_period=Li_configs['no_train_period']
        
    else:
        no_train_period=0
    if 'warmup_period' in Li_configs and Li_configs['warmup_period']>0:
        warmup_period=Li_configs['warmup_period']+no_train_period
    else:
        warmup_period=0
    
    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        entropy_ep=[]
        iters = len(ds.train)
        
        #[Li] warm up
        if Li_configs['li_flag']:
            if ep<no_train_period:
                optimizer_Li.param_groups[0]['lr']=0                            
            elif ep<warmup_period:
                optimizer_Li.param_groups[0]['lr']=Li_configs['lr']/Li_configs['warmup_period']*(ep-no_train_period+1)
            else:
                optimizer_Li.param_groups[0]['lr']=Li_configs['lr']
            
        #samples_memory=None#
        for n_iter, (samples, _) in enumerate(tqdm(ds.train, position=1)):
            #if not samples_memory:
            #    samples_memory=samples#!
            #else:
            #    samples=samples_memory#!
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            optimizer.zero_grad()
            
            if Li_configs['li_flag']: #[Li] This part should be improved
                batch_size=samples[0].shape[0]
                samples_concat=torch.cat(samples, dim=0)
                #samples_concat_origin=samples_concat#!
                samples_concat, logprob=Li(samples_concat.cuda())
                #samples_concat, logprob=Li(samples[0].cuda(), n_copies=2)
                samples=[samples_concat[:batch_size], samples_concat[batch_size:]]#! One of the sample is not cropped           
                
            
            loss = model(samples)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            model.step(ep / cfg.epoch)
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / iters)
                        
            if Li_configs['li_flag']: #[Li]
                model.eval()
                optimizer_Li.zero_grad() #Don't forget zero_grad
                loss_predictor = model(samples, mean=False) ##Is there a issue with it?
                loss_Li_pre=(loss_predictor.detach()*logprob).mean()+loss_predictor.mean() #!The last half
                
                entropy=Li.entropy()
                entropy_np=entropy.detach().cpu().numpy()
                
                if not start_entropy:
                    start_entropy=entropy_np[0]#!#!#!
                r=min(1, ep/Li_configs['entropy_increase_period'])
                if cfg.target_entropy>0:
                    target_entropy=cfg.target_entropy
                else:
                    target_entropy=Li_configs['target_entropy']
                
                mid_target_entropy=target_entropy*r+start_entropy*(1-r)
                loss_Li=loss_Li_pre+(entropy.mean()-mid_target_entropy)**2*0.3#!#!#!#!
                
                
                loss_Li.backward()
                optimizer_Li.step()
                entropy_ep.append(entropy_np)
                model.train()
                #for i in range(len(Li_configs['entropy_min_thresholds'])):
                #    entropy_step=Li.schedulers[i].step(entropy_np[i])
                #    Li_configs['entropy_weights'][i]*=entropy_step
                #    print(entropy_np[i], Li_configs['entropy_weights'][i])
        #[Li] Adjust entropy weights for diversity constraints.
        if Li_configs['li_flag']:
            entropy_ep=np.stack(entropy_ep, axis=0).mean(axis=0)
            entropy_dict={}
            for i in range(entropy_ep.shape[0]):
                entropy_dict['entropy_'+str(i)]=entropy_ep[i]
            wandb.log(entropy_dict, commit=False)
        
            #for i in range(len(Li_configs['entropy_min_thresholds'])):
            #    entropy_step=Li.schedulers[i].step(entropy_ep[i])
            #    Li_configs['entropy_weights'][i]*=entropy_step
            entropy_weight_dict={}
            for i in range(entropy_ep.shape[0]):
                entropy_weight_dict['entropy_weight_'+str(i)]=Li_configs['entropy_weights'][i]
            wandb.log(entropy_weight_dict, commit=False)
        
        
        if cfg.lr_step == "step":
            scheduler.step()

        if len(cfg.drop) and ep == (cfg.epoch - cfg.drop[0]):
            eval_every = cfg.eval_every_drop

        if (ep + 1) % eval_every == 0:
            acc_knn, acc = model.get_acc(ds.clf, ds.test, Li_configs=Li_configs, Li=Li)
            wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, commit=False)

        if (ep + 1) % cfg.eval_every == 0:
            fname = f"{cfg.model_folder}/{cfg.method}_{cfg.dataset}_{ep}.pt"
            torch.save(model.state_dict(), fname)
            fname_Li = f"{cfg.model_folder}/{cfg.method}_{cfg.dataset}_{ep}_Li.pt"
            torch.save(Li.augmentation.get_param.conv.state_dict(), fname_Li)

        wandb.log({"loss": np.mean(loss_ep), "ep": ep})
