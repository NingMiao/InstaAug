import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from .math_op import Generators, Crop
from .utils import RGB2HSV, HSV2RGB, normalize, denormalize
import time
from .distribution import Cropping_New_Param_Categorical_Dist_ConvFeature
import torchvision.transforms as transforms


class Augmentation_New_Param(nn.Module):
    """docstring for MLPAug """

    def __init__(self, conv, cfg={}, device='cuda'):
        super(Augmentation_New_Param, self).__init__()
        self.get_param = conv(bias=cfg['bias'], device=device).to(device)
        self.distC_crop_new_param=Cropping_New_Param_Categorical_Dist_ConvFeature( device=device).to(device)
        self.device=device
        if 'fixed_pretrain' in cfg:
            self.fixed_pretrain=cfg['fixed_pretrain']
        else:
            self.fixed_pretrain=None
        

                 
        
    def parameters(self):
        return self.get_param.parameters()

    def forward(self, x, n_copies=1, *args, **kwargs):
        
        #output_max=K is to output the top-K samples, which only works for cropping
        bs, _, w, h = x.size()
        
        logits = self.get_param(x, fixed_pretrain=self.fixed_pretrain)
        if self.fixed_pretrain is not None:
            if not hasattr(self, 'batch_size') or self.batch_size!=x.shape[0]:
                self.batch_size=x.shape[0]
                self.prob=np.concatenate([np.ones([121], dtype=np.float32) * self.fixed_pretrain[0]/121, 
                                          np.ones([100], dtype=np.float32) * self.fixed_pretrain[1]/100, 
                                          np.ones([16], dtype=np.float32) * self.fixed_pretrain[2]/16, 
                                          np.ones([1], dtype=np.float32) * self.fixed_pretrain[3]/1])
                self.prob=torch.tensor(self.prob).to(self.device)
                self.prob=torch.tile(torch.unsqueeze(self.prob, 0), [self.batch_size, 1])
                self.logprob=torch.log(self.prob)
            
            #Sample batch id
            rand = torch.empty_like(self.logprob).uniform_()
            samples = (-rand.log()+self.logprob).topk(k=1).indices[:,0]
            
            with torch.no_grad():
                x_out=self.get_param.get_image(samples)
                return x_out, x[:,0,0,0], x[:,0,0,0:1], x[:,0,0,0]
            
        samples, entropy_every, sample_logprob, KL_every = self.distC_crop_new_param(logits, n_copies=n_copies)
        sample_logprob = sample_logprob.transpose(0,1).reshape([-1])
        samples=samples.reshape([-1])
        x=torch.tile(x, [n_copies,1,1,1])
        with torch.no_grad():
            x_out=self.get_param.get_image(samples)
            
        return x_out, sample_logprob, entropy_every, KL_every
        
        
        
if __name__=='__main__':
    pass