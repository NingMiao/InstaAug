from .augmentation import Augmentation 
from .augmentation_new_param import Augmentation_New_Param
from .augmentation_out_source_param import Augmentation_Out_Source_Param
from .architectures.mlp import Mlp
from .architectures import PreActResNet, SimpleConv_tiny_imagenet, PreActResFeatureNet, PreActResFeatureNet_Rawfoot, PreActResFeatureNet_Imagenet, PreActResFeatureNet_Cifar, ImageLogit

import torch
import torch.nn as nn

from .scheduler import *

#Log: the api for entropy is changed from a separate function to one of the output of forward

class learnable_invariance(nn.Module):
    def __init__(self, cfg, device='cuda', tta_only=False):
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
            elif tta_only:
                FeatureNet=None
            else:
                FeatureNet=PreActResFeatureNet_Imagenet
        elif 'cifar' in cfg['dataset']:
            FeatureNet=PreActResFeatureNet_Cifar
        
        if self.mode=='crop_new_param' and 'imagenet' == cfg['dataset']:
            self.augmentation=Augmentation_New_Param(FeatureNet, cfg, device=device)
        elif tta_only:
            self.augmentation=Augmentation_Out_Source_Param(cfg=cfg, tta_only=tta_only)
        else:
            self.augmentation=Augmentation(FeatureNet, cfg['transform'], cfg, tta_only=tta_only)   
        self.__init__scheduler()
        self.tta_only=tta_only
    
    def parameters(self):
        return self.augmentation.parameters()
    
    def forward(self, x, n_copies=1, output_max=0, tta_train=True, params=None, random_crop_aug=False):
        #input x has shape [batch_size, 3, h, w]
        #outputs=x_transformed([batch_size, 3, h, w]), logprob([batch_size])
        random_crop_aug=random_crop_aug
        random_color_aug=False
        hsv_input=True if self.cfg['mode']=='color' else False
        
        if self.cfg['random_aug']:
            if self.cfg['mode']=='crop':
                random_crop_aug=True
            elif self.cfg['mode']=='color':
                random_color_aug=True
        
        return self.augmentation(x, n_copies=n_copies, output_max=output_max, global_aug=self.cfg['global_aug'], random_crop_aug=random_crop_aug, random_color_aug=random_color_aug, hsv_input=hsv_input, tta_train=tta_train, params=params)
    
    def __init__scheduler(self):
        self.schedulers=[]
        for i in range(len(self.cfg['entropy_max_thresholds'])):
            self.schedulers.append(Scheduler(self.cfg['entropy_min_thresholds'][i],self.cfg['entropy_max_thresholds'][i]))
        
if __name__=='__main__':
    torch.autograd.set_detect_anomaly(True)
    
    flag=0
    
    if flag==0:
        import yaml
        config_path='./InstaAug_module/configs/config_crop_supervised.yaml'
        Li_configs = yaml.safe_load(open(config_path,'r'))
        Li=learnable_invariance(Li_configs) 
            
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
        config_path='./InstaAug_module/configs/config_crop_supervised_imagenet.yaml'
        Li_configs = yaml.safe_load(open(config_path,'r'))
        Li=learnable_invariance(Li_configs, device='cuda', tta_only=True) 
        
        x=torch.normal(0,1,size=[2, 3, 224, 224]).to(torch.float32).cuda()
        #x=torch.randn([1, 3, 224, 224], dtype=torch.float32)
        #x[:,:,::5, ::10]=1
        
        #x_out, sample_logprob, entropy_every, KL_every=Li(x, 10)
        for i in range(3000):
            x_out, sample_logit, sample_id=Li(x, 10, tta_train=True)
            loss = (-sample_logit[0]*((sample_id[0]>50).to(sample_id.dtype)-1)).mean()+(-sample_logit[1]*((sample_id[1]<50).to(sample_id.dtype)-1)).mean()
            loss=loss.mean()
            loss.backward()
            for param in Li.parameters():
                param.data-=0.000001*param.grad
            if i%10==0:
                print(loss)
                if loss<-1000:
                    break
        
        x_out, sample_logit, sample_id=Li(x, 10, tta_train=False)
        print(x_out.shape, sample_logit.shape, sample_id.shape)
        print(sample_id)
        print(sample_logit)
        
        
        #print(x_out.shape, sample_logprob.shape, entropy_every.shape, KL_every.shape)
        #print(entropy_every)
        #print(Li.augmentation.get_param.idx2detail)
        
        #print(Li.augmentation.get_param.get_image(torch.tensor([1, 121, 221, 237], dtype=torch.int32)))
        
        #import numpy as np
        #np.save('x_out.npy', x_out.numpy())
        #np.save('x.npy', x.numpy())
    elif flag==2:
        import yaml
        config_path='./InstaAug_module/configs/config_crop_supervised_imagenet.yaml'
        Li_configs = yaml.safe_load(open(config_path,'r'))
        Li=learnable_invariance(Li_configs, device='cpu', tta_only=True)
        
        x=torch.normal(0,1,size=[2, 3, 224, 224]).to(torch.float32).cuda()
        Li=nn.DataParallel(Li)
        Li.cuda()
        print(Li(x, 10, tta_train=True))
        
    elif flag==3:
        import yaml
        config_path='./InstaAug_module/configs/config_crop_imagenet_tta_only.yaml'
        Li_configs = yaml.safe_load(open(config_path,'r'))
        Li=learnable_invariance(Li_configs, device='cpu', tta_only=True)
        #x=torch.normal(0,1,size=[2, 1, 224, 224]).to(torch.float32)
        import numpy as np
        x=np.tile(np.arange(0, 224).reshape([1, -1]).T, [224, 1]).reshape([1,1,224,-1])
        x=np.tile(x,[1,3,1,1])
        x=torch.tensor(x).to(torch.float32)
        y=x[:,1].transpose(1,2).clone()
        x[:,1]=y
        params=torch.normal(0,1,size=[1, 30]).to(torch.float32)
        out=Li(x, n_copies=10, tta_train=False, params=params)[0]
        #print(out[0,0,0], out[0,0,:,0])
        print(out[1,0,0], out[1,1,:,0])
        #print(x[0,0,0], x[0,1,:,0])
        
    