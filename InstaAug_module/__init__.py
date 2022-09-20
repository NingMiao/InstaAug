from .augmentation import Augmentation 
from .augmentation_new_param import Augmentation_New_Param
from .architectures.mlp import Mlp
from .architectures import PreActResNet, SimpleConv_tiny_imagenet, PreActResFeatureNet, PreActResFeatureNet_Rawfoot, PreActResFeatureNet_Imagenet, PreActResFeatureNet_Cifar, ImageLogit

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
            if self.mode=='crop_new_param':
                FeatureNet=ImageLogit
            else:
                FeatureNet=PreActResFeatureNet_Imagenet
        elif 'cifar' in cfg['dataset']:
            FeatureNet=PreActResFeatureNet_Cifar
        
        if self.mode=='crop_new_param':
            self.augmentation=Augmentation_New_Param(FeatureNet, cfg, device=device)
        else:
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
        
if __name__=='__main__':
    torch.autograd.set_detect_anomaly(True)
    
    flag=1
    
    if flag==0:
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
    elif flag==1:
        import yaml
        Li_configs = yaml.safe_load(open('./InstaAug_module/configs/config_crop_supervised_imagenet_new_param.yaml','r'))
        Li=learnable_invariance(Li_configs, device='cpu') 
        
        x=torch.zeros([1, 3, 224, 224], dtype=torch.float32)
        x=torch.randn([1, 3, 224, 224], dtype=torch.float32)
        #x[:,:,::5, ::10]=1
        
        x_out, sample_logprob, entropy_every, KL_every=Li(x, 10)
        #print(x_out.shape, sample_logprob.shape, entropy_every.shape, KL_every.shape)
        #print(entropy_every)
        #print(Li.augmentation.get_param.idx2detail)
        
        print(Li.augmentation.get_param.get_image(torch.tensor([1, 121, 221, 237], dtype=torch.int32)))
        
        #import numpy as np
        #np.save('x_out.npy', x_out.numpy())
        #np.save('x.npy', x.numpy())

    
    