#from .augmentation import AmortizedParamDist, Cropping_Uniform_Dist, Augmentation
#from .augmentation_ConvFeature import Augmentation_ConvFeature
from .augmentation_ConvFeature import Augmentation_ConvFeature 
from .architectures.mlp import Mlp
from .architectures import PreActResNet, SimpleConv_tiny_imagenet, PreActResFeatureNet, PreActResFeatureNet_Rawfoot, PreActResFeatureNet_Imagenet

import torch
import torch.nn as nn

from .scheduler import *

mode='crop'
dataset='tiny'
#mode='color'
#dataset='raw'
#mode='crop_meanfield'
#dataset='tiny'

if 'tiny' in dataset:
    PreActResFeatureNet=PreActResFeatureNet
elif 'raw' in dataset:
    PreActResFeatureNet=PreActResFeatureNet_Rawfoot
elif 'marioiggy' in dataset:
    PreActResFeatureNet=PreActResFeatureNet

class cfg_crop:
    transform=['crop']    
    random_aug=False#!
    global_aug=False#!    

class cfg_color:
    transform=['h','s','v']#!
    random_aug=False
    global_aug=False

class cfg_color_and_crop:
    transform=['h','s','v', 'crop']   
    random_aug=False
    global_aug=False

class cfg_rotation:
    transform=['rotation']   
    random_aug=False
    global_aug=False

class cfg_crop_meanfield:
    transform=['crop_meanfield']   
    random_aug=False
    global_aug=False    

class aug_param_crop:
    #Should warm up if training from scratch
    li_flag=False #!
    load_li=False #!
    train_copies=1 #!
    test_time_aug=True
    test_copies=50 #!#trial
    lr=1e-5#!Careful
    #lr=3e-7 #!For finetune only
    warmup_period=0
    
    #crop_entropy_weight=0.3#!For cropping only    
    #crop_max_threshold= 5
    #crop_min_threshold= 4
    
    crop_entropy_weight=0.2#!For supervised
    #crop_entropy_weight=0.03#!For contrastive? 
    
    #Be careful of the order of min and max
    crop_max_threshold= 4.0
    crop_min_threshold= 3.0
    
    scheduler_sleep_epoch=0 #!
    ConvFeature=True
    max_tolerance=30   #! For supervised, choose 30, for random supervised, choose 10
    #!max_tolerance=100#! For contrastive only     
    
class aug_param_color:
    li_flag=True #
    load_li=False #!
    train_copies=1 #!
    test_time_aug=True
    test_copies=10 #!
    lr=1e-5 #!Careful
    warmup_period=0
    
    color_entropy_weights=[0.3, 0.3, 0.1, 0.1, 0.1, 0.3]
    #color_max_thresholds=[0.05,0.05,1.0,1.0,1.0,1.0]
    #color_min_thresholds=[0.0, 0.0, 0.0,0.0,0.8,0.8]
    color_max_thresholds=[1.0,1.0,1.0,1.0,1.0,1.0]
    color_min_thresholds=[0.0, 0.0, 0.0,0.0,0.6,0.6]
    
    scheduler_sleep_epoch=0
    ConvFeature=True
    max_tolerance=10
    

class aug_param_color_and_crop:
    li_flag=False #!
    load_li=False #!
    train_copies=1 #!
    test_time_aug=True
    test_copies=10 #!
    lr=1e-5 #!Careful
    warmup_period=10
    
    color_entropy_weights=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1] #Be careful, this weight is different from old version
    color_max_thresholds=[0.6,0.6,1.5,1.5,2.0,2.0]
    color_min_thresholds=[0.0,0.0,0.0,0.0,0.00,0.0]
    
    crop_entropy_weight=0.4#!For cropping only
    crop_max_threshold= 100.0
    crop_min_threshold= 0.0
    
    scheduler_sleep_epoch=0
    ConvFeature=True
    max_tolerance=10    

    
class aug_param_rotation:
    li_flag=True #!
    load_li=True #!
    train_copies=1 #!
    test_time_aug=False
    test_copies=10 #!
    lr=1e-5 #!Careful
    warmup_period=10
    
    color_entropy_weights=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1] #Be careful, this weight is different from old version
    color_max_thresholds=[0.6,0.6,1.5,1.5,2.0,2.0]
    color_min_thresholds=[0.0,0.0,0.0,0.0,0.00,0.0]
    
    crop_entropy_weight=0.4#!For cropping only
    crop_max_threshold= 100.0
    crop_min_threshold= 0.0
    
    rotation_entropy_weights=[0.1]
    
    scheduler_sleep_epoch=0
    ConvFeature=True
    max_tolerance=1000      

class aug_param_crop_meanfield:
    li_flag=True #!
    load_li=True #!
    train_copies=1 #!
    test_time_aug=True
    test_copies=50 #!
    lr=1e-5 #!Careful
    warmup_period=0 #For all param?
    
    crop_meanfield_entropy_weights=[0.01]
    
    scheduler_sleep_epoch=0
    ConvFeature=True
    max_tolerance=30       
    
if mode=='crop':
    aug_param=aug_param_crop
    cfg=cfg_crop
elif mode=='color':
    aug_param=aug_param_color
    cfg=cfg_color
elif mode=='color_and_crop':
    aug_param=aug_param_color_and_crop
    cfg=cfg_color_and_crop
elif mode=='rotation':
    aug_param=aug_param_rotation
    cfg=cfg_rotation
elif mode=='crop_meanfield':
    aug_param=aug_param_crop_meanfield
    cfg=cfg_crop_meanfield
    
aug_param.mode=mode


class learnable_invariance(nn.Module):
    def __init__(self, cfg):
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
        
        
        self.augmentation=Augmentation_ConvFeature(FeatureNet, cfg['transform'], cfg)   
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
    
    def entropy(self, mean=True):
        #outputshape=[batch_size, aug_dimension], which is different from the old setting for crop
        if self.cfg['mode']=='crop':
            return self.crop_entropy(mean=mean)
        elif self.cfg['mode']=='color':
            return self.color_entropy(mean=mean)
        elif self.cfg['mode']=='color_and_crop':
            color_entropy=self.color_entropy(mean=mean)
            crop_entropy=self.crop_entropy(mean=mean)
            return torch.cat([color_entropy, crop_entropy], dim=1)
        elif self.cfg['mode']=='rotation':
            return self.rotation_entropy(mean=mean)
        elif self.cfg['mode']=='crop_meanfield':
            return self.crop_meanfield_entropy(mean=mean)
    
    
    def crop_entropy(self, mean=True):
        #entropy_every has shape [batch_size]
        if not hasattr(self.augmentation, 'dist_crop'):
            return None
        if mean:
            return self.augmentation.dist_crop.entropy_every.mean(dim=0)
        else:
            return self.augmentation.dist_crop.entropy_every
    
    def color_entropy(self, mean=True):
        #entropy_every has shape [batch_size, #transforms*2]
        if not hasattr(self.augmentation, 'dist_color'):
            return None
        
        if mean:
            return self.augmentation.dist_color.entropy_every.mean(dim=0)
        else:
            return self.augmentation.dist_color.entropy_every
    
    def rotation_entropy(self, mean=True):
        if not hasattr(self.augmentation, 'dist_rotation'):
            return None
        
        if mean:
            return self.augmentation.dist_rotation.entropy_every.mean(dim=0)
        else:
            return self.augmentation.dist_rotation.entropy_every
    
    def crop_meanfield_entropy(self, mean=True):
        #entropy_every has shape [batch_size, #transforms*2]
        if not hasattr(self.augmentation, 'dist_crop_meanfield'):
            return None
        
        if mean:
            return self.augmentation.dist_crop_meanfield.entropy_every.mean(dim=0)
        else:
            return self.augmentation.dist_crop_meanfield.entropy_every
    
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
    
    