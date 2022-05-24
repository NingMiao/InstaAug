import torch 
import numpy as np
from torchdiffeq import odeint
import time

def MoG_entropy_0(params_pos, sample_num_component=1, ranges=[1.6, 1.6]):
    #estimate entropy of MoG.
    sample_num=params_pos.shape[1]
    dim=params_pos.shape[2]
    std=1.0/(sample_num)**0.5
    sample_mean=params_pos.unsqueeze(2).expand(-1,-1, sample_num_component, -1)
    sample_r=(torch.randn(sample_mean.shape)*std).type_as(sample_mean)*0#!Temporary setting
    sample_MoG = sample_mean+sample_r
    sample_MoG = sample_MoG.view(sample_MoG.shape[0], -1,1, sample_MoG.shape[-1])
    sample_sub=sample_MoG-params_pos.unsqueeze(1)
    sample_sub_abs=torch.abs(sample_sub)
    range_tensor=torch.tensor(ranges).type_as(sample_sub_abs)
    
    for i in range(len(sample_sub_abs.shape)-1):
        range_tensor.unsqueeze(0)
    sample_sub_quotient=torch.minimum(sample_sub_abs, range_tensor-sample_sub_abs)
    
    #!tem_1=1/((2*np.pi)**0.5*std)**dim*torch.exp(-(sample_sub_quotient**2).sum(axis=-1)/(2*std**2))
    
    beta=3 #beta is the parameter that defines the decay of loss wrt distance
    tem_1=torch.exp(-beta*((sample_sub_quotient**2).sum(axis=-1)+1e-5)**0.5) #~ n1 distance
    sample_prob=(tem_1*torch.unsqueeze(1-torch.eye(tem_1.shape[-1]).type_as(sample_sub_abs), 0)).sum(axis=-1)/(tem_1.shape[-1]-1)
    #sample_prob_log=torch.log(sample_prob) #?
    #entropy=-sample_prob_log.mean(axis=-1) #?
    entropy=-sample_prob.mean(axis=-1)
    return entropy

def MoG_entropy_1(params_pos, sample_num_component=1, ranges=[1.6, 1.6], beta=10):
    #estimate entropy of MoG.
    sample_num=params_pos.shape[1]
    dim=params_pos.shape[2]
    sample_mean=params_pos.unsqueeze(2)
    sample_mean_T = sample_mean.view(sample_mean.shape[0], 1,-1, sample_mean.shape[-1])
    sample_sub=sample_mean-sample_mean_T
    sample_sub_abs=torch.abs(sample_sub)
    range_tensor=torch.tensor(ranges).type_as(sample_sub_abs)
    
    #for i in range(len(sample_sub_abs.shape)-1):
    #    range_tensor.unsqueeze(0)
    #sample_sub_quotient=torch.minimum(sample_sub_abs, range_tensor-sample_sub_abs)
    sample_sub_quotient=sample_sub_abs
    #beta is the parameter that defines the decay of loss wrt distance
    tem_1=torch.exp(-beta*((sample_sub_quotient**2).sum(axis=-1)+1e-5)**0.5) #~ n1 distance
    sample_prob=(tem_1*torch.unsqueeze(1-torch.eye(tem_1.shape[-1]).type_as(sample_sub_abs), 0)).sum(axis=-1)/(tem_1.shape[-1]-1)
    #entropy=-sample_prob.mean(axis=-1)
    entropy=-torch.log(sample_prob).mean(axis=-1) #? Use log or not
    return entropy

MoG_entropy=MoG_entropy_1 #!

def expm(A, rtol=1e-4):
    """assumes A has shape (bs,d,d)
    returns exp(A) with shape (bs,d,d)"""
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)[None].repeat(A.shape[0], 1, 1)
    return odeint(lambda t, x: A @ x, I, torch.tensor([0.0, 1.0]).to(A.device, A.dtype), rtol=rtol)[-1]


def RGB2HSV(img):
    img_max, img_max_channel=torch.max(img, dim=-1)    
    img_min, img_min_channel=torch.min(img, dim=-1)
    img_mid=torch.sum(img, dim=-1)-img_max-img_min
    
    L=img_max
    c=img_max-img_min
    S=c/(L+1e-7)
    
    H_weight=(img_mid-img_min)/(c+1e-7)
    
    H_center_index=img_max_channel
    H_center=H_center_index/3
    
    H_direction_index=torch.remainder(img_max_channel-img_min_channel, 3)-1
    H_direction=-(H_direction_index*2-1)
    
    H=H_center+1/6*H_direction*H_weight
    
    
    return torch.stack([H, S, L], dim=-1) ##Adding img*0 make the 
    
    
    
def HSV2RGB(img):    
    H, S, L=img[...,0], img[...,1], img[...,2]
    large_index=torch.floor(torch.remainder((H+1/6)*3, 3)).type(torch.int64)
    mid_index_pre=torch.floor(torch.remainder(-(H+1/6)*6, 2))+1
    mid_index=torch.remainder(large_index+mid_index_pre, 3).type(torch.int64)
    small_index=3-large_index-mid_index
    
    boundary=torch.abs(torch.remainder(H*3+0.5, 1)-0.5)*2
    
    large_value=L
    small_value=L*(1-S)
    mid_value=L*(1-S*(1-boundary))
    
    RGB_large=torch.nn.functional.one_hot(large_index, 3)*large_value.unsqueeze(-1)
    RGB_mid=torch.nn.functional.one_hot(mid_index, 3)*mid_value.unsqueeze(-1)
    RGB_small=torch.nn.functional.one_hot(small_index, 3)*small_value.unsqueeze(-1)
    
    RGB=RGB_large+RGB_mid+RGB_small
    return RGB

def normalize(x, means, stds, dim=1):
    means=torch.tensor(means).unsqueeze(0).unsqueeze(2).unsqueeze(3).type(x.type())
    stds=torch.tensor(stds).unsqueeze(0).unsqueeze(2).unsqueeze(3).type(x.type())
    return (x-means)/stds
def denormalize(x, means, stds, dim=1):
    #Time consuming
    t=time.time()
    print(means)
    means=torch.tensor(means).unsqueeze(0).unsqueeze(2).unsqueeze(3).type(x.type())
    print('step 0.20', str(time.time()-t))#!
    t=time.time()
    stds=torch.tensor(stds).unsqueeze(0).unsqueeze(2).unsqueeze(3).type(x.type())
    print('step 0.21', str(time.time()-t))#!
    t=time.time()
    out=x*stds+means
    print('step 0.22', str(time.time()-t))#!
    t=time.time()
    return out

if __name__=='__main__':
    import torch.nn as nn
    #A=np.random.random([1,20,2])*1.0
    #A[:,:,0]=A[:,:,0]*1.6-0.8
    #A[:,:,1]=A[:,:,1]*0.0001
    #A=np.array([[[0,0],[0.25,0.01],[0.5, 0.0]],[[0.0,0.0],[0.02,0.3],[0.2, 0.0]]])
    A=np.array([[[0,0],[0.0,0.0],[0.0, 0.0]],[[0.8,0.8],[0.8,0.8],[0.8, 0.8]]])
    param=nn.Parameter(torch.tensor(A))
    loss=MoG_entropy(param)
    #loss=param.sum()
    loss.sum().backward()
    print(param.grad)
    
    print(loss)