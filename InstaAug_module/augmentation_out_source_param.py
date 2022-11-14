import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from .math_op import Generators, Crop
from .utils import RGB2HSV, HSV2RGB, normalize, denormalize
import time
from .distribution import Cropping_Categorical_Dist_ConvFeature, Color_Uniform_Dist_ConvFeature, Rotation_Uniform_Dist_ConvFeature, Crop_Meanfield_Uniform_Dist_ConvFeature, Cropping_Categorical_tta_only_Dist_ConvFeature

import torchvision.transforms as transforms#!
#!

        
class Augmentation_Out_Source_Param(nn.Module):
    #Use a outside network to provide parameters for augmentatation. Supports crop only. Supports tta only.

    def __init__(self, transform='crop', cfg={}, device='cuda', tta_only=False):
        #Not need to have a conv here
        super(Augmentation_Out_Source_Param, self).__init__()
        self.transform = transform   
        
        sizes=cfg['sizes']
        scopes_crop=sizes
        nums=cfg['nums']
        
        centers_crop=[]
        for i in range(len(sizes)):
            centers_crop_layer=[]
            size=sizes[i]
            num=nums[i]
            if num==1:
                center_interval=0
            else:
                center_interval=(1-size)/(num-1)*2
            center_lower=size-1
            for j1 in range(num):
                for j2 in range(num):
                    x1=center_lower+center_interval*j1
                    x2=center_lower+center_interval*j2
                    centers_crop_layer.append([x1, x2])
            centers_crop.append(centers_crop_layer)
        
        centers_crop=[torch.tensor(item) for item in centers_crop]
        center_intervals_crop=[0.0 for i in sizes]
        scope_ranges_crop=[[0.0, 0.0] for i in sizes]                
        
        if not tta_only:
            self.distC_crop=Cropping_Categorical_Dist_ConvFeature(centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop)
        else:
            self.distC_crop=Cropping_Categorical_tta_only_Dist_ConvFeature(centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop)
        
        
         
        
    def parameters(self):
        return []
    
    def forward(self, x, n_copies=1, tta_train=True, params=None, **kwargs):
        return self.forward_tta_only(x, n_copies=n_copies, tta_train=tta_train, params=params)

        
    def forward_tta_only(self, x, n_copies=1, tta_train=True, params=None, **kwargs):
        #tta only mode, only supports cropping now.
        
        bs, _, w, h = x.size()
                
        x_input=x
        x=torch.tile(x,[n_copies, 1,1,1])
        
        params_crop = params
        
        #Crop only
        if 'crop' in self.transform:            
            #Sample transformation
            weights, sample_logit, sample_id = self.distC_crop(params_crop, n_copies=n_copies, avoid_black_margin=False, output_max=False, tta_train=tta_train)# sample_id is the position of patches
            weights = weights.transpose(0,1).reshape([-1, weights.shape[2]]) #shape=[batch_size*n_copies, para_dim]
            #sample_logprob = sample_logprob.transpose(0,1).reshape([-1])  #shape=[batch_size*n_copies]
            
           
            ## exponential map to apply transformations
            i=0
            for tranformation in ['crop']:
                weights_dim=Generators.weights_dim(tranformation)
                mat = getattr(Generators, tranformation)(weights[:,i:i+weights_dim])
                if i==0:
                    affine_matrices=mat
                else:
                    affine_matrices=torch.matmul(affine_matrices, mat)
                    
                i+=weights_dim
            

                
            flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(), align_corners=True)
            

            
            x_out = F.grid_sample(x, flowgrid, align_corners=True)
            
            return x_out, sample_logit, sample_id
        else:
            return x, torch.zeros([x.shape[0]]).to(x.device())   
        
if __name__=='__main__':
    pass