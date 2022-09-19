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
        

                 
        
    def parameters(self):
        return self.get_param.parameters()

    def forward(self, x, n_copies=1, *args, **kwargs):
        
        #output_max=K is to output the top-K samples, which only works for cropping
        bs, _, w, h = x.size()
        
        logits = self.get_param(x)
        samples, entropy_every, sample_logprob, KL_every = self.distC_crop_new_param(logits, n_copies=n_copies)
        sample_logprob = sample_logprob.transpose(0,1).reshape([-1])
        samples=samples.reshape([-1])
        x=torch.tile(x, [n_copies,1,1,1])
        x_out=self.get_param.get_image(samples)
        
            
        return x_out, sample_logprob, entropy_every, KL_every
        
        
        
if __name__=='__main__':
    pass