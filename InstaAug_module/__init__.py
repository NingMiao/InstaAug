from .augmentation import Augmentation 
from .architectures.mlp import Mlp
from .architectures import PreActResNet, SimpleConv_tiny_imagenet, PreActResFeatureNet, PreActResFeatureNet_Rawfoot, PreActResFeatureNet_Imagenet, PreActResFeatureNet_Cifar

import torch
import torch.nn as nn

from .scheduler import *

#Log: the api for entropy is changed from a separate function to one of the output of forward

class learnable_invariance(nn.Module):
    def __init__(self, cfg, device='cuda'):
        #numinputs: 2:32768, 4:8192, top: 512
        super(learnable_invariance, self).__init__()
        self.cfg=cfg
        self.mode=cfg['mode']
        
        if 'tiny' in cfg['dataset']:
            FeatureNet=PreActResFeatureNet
        elif 'raw' in cfg['dataset']:
            FeatureNet=PreActResFeatureNet_Rawfoot
        elif 'marioiggy' in cfg['dataset']:
            FeatureNet=PreActResFeatureNet
        elif 'imagenet' == cfg['dataset']:
            FeatureNet=PreActResFeatureNet_Imagenet
        elif 'cifar' in cfg['dataset']:
            FeatureNet=PreActResFeatureNet_Cifar
        
        
        self.augmentation=Augmentation(FeatureNet, cfg['transform'], cfg, device=device)   
        self.__init__scheduler()
    
    def parameters(self):
        return self.augmentation.parameters()
    
    def forward(self, x, n_copies=1, output_max=0):
        #input x has shape [batch_size, 3, h, w]
        #outputs=x_transformed([batch_size, 3, h, w]), logprob([batch_size])
        random_crop_aug=False
        random_color_aug=False
        hsv_input=True if self.cfg['mode']=='color' else False
        
        if self.cfg['random_aug']:
            if self.cfg['mode']=='crop':
                random_crop_aug=True
            elif self.cfg['mode']=='color':
                random_color_aug=True
        
        return self.augmentation(x, n_copies=n_copies, output_max=output_max, global_aug=self.cfg['global_aug'], random_crop_aug=random_crop_aug, random_color_aug=random_color_aug, hsv_input=hsv_input)
    
    def __init__scheduler(self):
        self.schedulers=[]
        for i in range(len(self.cfg['entropy_max_thresholds'])):
            self.schedulers.append(Scheduler(self.cfg['entropy_min_thresholds'][i],self.cfg['entropy_max_thresholds'][i]))
    
    #def entropy(self, mean=True):
    #    #outputshape=[batch_size, aug_dimension], which is different from the old setting for crop
    #    if self.cfg['mode']=='crop':
    #        return self.crop_entropy(mean=mean)
    #    elif self.cfg['mode']=='color':
    #        return self.color_entropy(mean=mean)
    #    elif self.cfg['mode']=='color_and_crop':
    #        color_entropy=self.color_entropy(mean=mean)
    #        crop_entropy=self.crop_entropy(mean=mean)
    #        return torch.cat([color_entropy, crop_entropy], dim=1)
    #    elif self.cfg['mode']=='rotation':
    #        return self.rotation_entropy(mean=mean)
    #    elif self.cfg['mode']=='crop_meanfield':
    #        return self.crop_meanfield_entropy(mean=mean)
    
    
    #def crop_entropy(self, mean=True):
    #    #entropy_every has shape [batch_size]
    #    if not hasattr(self.augmentation, 'dist_crop'):
    #        return None
    #    if mean:
    #        return self.augmentation.dist_crop.entropy_every.mean(dim=0)
    #    else:
    #        return self.augmentation.dist_crop.entropy_every
    
    #def color_entropy(self, mean=True):
    #    #entropy_every has shape [batch_size, #transforms*2]
    #    if not hasattr(self.augmentation, 'dist_color'):
    #        return None
    #    
    #    if mean:
    #        return self.augmentation.dist_color.entropy_every.mean(dim=0)
    #    else:
    #        return self.augmentation.dist_color.entropy_every
    
    #def rotation_entropy(self, mean=True):
    #    if not hasattr(self.augmentation, 'dist_rotation'):
    #        return None
    #    
    #    if mean:
    #        return self.augmentation.dist_rotation.entropy_every.mean(dim=0)
    #    else:
    #        return self.augmentation.dist_rotation.entropy_every
    
    #def crop_meanfield_entropy(self, mean=True):
    #    #entropy_every has shape [batch_size, #transforms*2]
    #    if not hasattr(self.augmentation, 'dist_crop_meanfield'):
    #        return None
    #    
    #    if mean:
    #        return self.augmentation.dist_crop_meanfield.entropy_every.mean(dim=0)
    #    else:
    #        return self.augmentation.dist_crop_meanfield.entropy_every
    
if __name__=='__main__':
    torch.autograd.set_detect_anomaly(True)
    
    Li=learnable_invariance()
    
    x=torch.randn([17, 3, 64,64])
    for i in range(100):
        
        ds, logprob=Li(x)
        print(ds.shape, logprob)
        if mode=='color':
            loss=-Li.color_entropy().mean()
        elif mode=='crop':
            loss=-Li.crop_entropy()
        elif mode=='rotation':
            loss=-Li.rotation_entropy().mean()
        else:
            loss=-Li.color_entropy().mean()-Li.crop_entropy()
        
        print(i, loss)
        loss.backward()
        for param in Li.parameters():
            param.data-=0.001*param.grad
    
    