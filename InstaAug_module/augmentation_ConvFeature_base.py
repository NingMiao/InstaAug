from .augmentation_base import *
#!torch.autograd.set_detect_anomaly(True)
import numpy as np


class AmortizedParamDist_ConvFeature(nn.Module):#Not in use
    """Parameters are a learnt from an amortized netork

    Args:
        shape_param (tuple): Shape of the parameters to be output
    """

    def __init__(self, conv, *args, **kwargs):#!
        super(AmortizedParamDist_ConvFeature, self).__init__()
        #self.conv=conv(output_layer=[2, 3,-1])  #!
        #self.bias=[0, 1, 3] #!
        self.conv=conv(output_layer=[3,-1])  #!
        self.bias=[1, 3] #!
        #Don't change this, not working now
        
    def parameters(self):
        return self.conv.parameters()

    def forward(self, x):
        
        params=self.conv(x)
        centers=self.conv.center()
        scopes=self.conv.scope()
        
        log_nums=[np.log(c.shape[0]*c.shape[0]) for c in centers]
        params=[params[i]+self.bias[i]-log_nums[i] for i in range(len(params))]
        
        return params, centers, scopes #~Leave activation to distribution   

class Cropping_Uniform_Dist_ConvFeature(ParametricDistribution):
    """ U([-theta, theta]) """
    #For our categorical cropping.
    len_param = 1

    def __init__(self, params, centers, center_intervals, sizes, size_ranges, num=1, n_copies=1, avoid_black_margin=False, smooth=True, trial=False, output_max=0, **kwargs):
        super(Cropping_Uniform_Dist_ConvFeature, self).__init__()
        
        param_list=[]
        center_list=[]
        size_list=[]
        
        center_interval_list=[]
        size_range_list=[]
        for i in range(len(params)):
            param=params[i]
            center=centers[i]
            center_interval=center_intervals[i]
            size=sizes[i]
            size_range=size_ranges[i]
            
            shape=param.shape
            param=param.view(shape[0], -1)
            param_list.append(param)
            
            center=center.reshape(-1, 2).to(param.device)
            center_list.append(center)
            
            zero_base=torch.zeros(center.shape).type(center.type())
            
            center_interval_list.append(zero_base+center_interval)
            
            size_list.append(zero_base+size)
            size_range_new=torch.stack([zero_base[:,0]+size_range[0], zero_base[:,1]+size_range[1]], dim=-1)
            size_range_list.append(size_range_new)
            
        
        params=torch.cat(param_list, dim=1) #Shape:[batch, num_patches]
        centers=torch.cat(center_list, dim=0) #Shape:[num_patches, 2]
        center_intervals=torch.cat(center_interval_list, dim=0) #Shape:[num_patches, 2]
        sizes=torch.cat(size_list, dim=0) #Shape:[num_patches, 2]
        size_ranges=torch.cat(size_range_list, dim=0) #Shape:[num_patches, 2]
        
        
        logprob=torch.nn.functional.log_softmax(params, dim=-1)
        prob=torch.exp(logprob)
        
        #print(prob[:,:100].sum(axis=1)[:5], prob[:,100:104].sum(axis=1)[:5], prob[:,104:].sum(axis=1)[:5])
        #print(prob[:,100:104][:5])
        #print(size_ranges[99:])
        
        if False:#!
            import numpy as np
            np.save('analysis/tem/prob.npy', prob.detach().cpu().numpy())
        
        samples=torch.multinomial(prob, num, replacement=True)
        
        
        if trial:#!#trial
            nums=prob.shape[1]#!
            logprob=torch.tile(logprob[:prob.shape[0]//n_copies],[nums, 1])#!For test only
            prob=torch.tile(prob[:prob.shape[0]//n_copies],[nums, 1])#!For test only
            samples=torch.arange(nums).unsqueeze(1).tile([1,prob.shape[0]//n_copies]).reshape([-1,1]).type(samples.type())#!
        if output_max>0:#!#output_max
            smooth=False
            nums=output_max#!
            logprob_single=logprob[:prob.shape[0]//n_copies]
            samples=torch.argsort(-logprob_single, axis=1)[:, :nums].transpose(1,0).reshape([-1,1])
            
            logprob=torch.tile(logprob[:prob.shape[0]//n_copies],[nums, 1])#!For test only
            prob=torch.tile(prob[:prob.shape[0]//n_copies],[nums, 1])#!For test only
            
        
        
        
        param_pos_list=[]
        param_pos_interval_list=[]
        param_size_list=[]
        param_size_range_list=[]
        for i in range(num):
            samples_reshape=samples[:, i].reshape(-1)
            param_pos=torch.index_select(centers, 0, samples_reshape)
            param_pos=param_pos.reshape(samples.shape[0], 1, -1)
            param_pos_list.append(param_pos)
            param_pos_interval=torch.index_select(center_intervals, 0, samples_reshape)
            param_pos_interval=param_pos_interval.reshape(samples.shape[0], 1, -1)
            param_pos_interval_list.append(param_pos_interval)
            
            param_size=torch.index_select(sizes, 0, samples_reshape)
            param_size=param_size.reshape(samples.shape[0], 1, -1)
            param_size_list.append(param_size)
            param_size_range=torch.index_select(size_ranges, 0, samples_reshape)
            param_size_range=param_size_range.reshape(samples.shape[0], 1, -1)
            param_size_range_list.append(param_size_range)
            
            
        param_pos=torch.cat(param_pos_list, dim=1)
        param_pos_interval=torch.cat(param_pos_interval_list, dim=1)
        param_size=torch.cat(param_size_list, dim=1)
        param_size_range=torch.cat(param_size_range_list, dim=1)
        
                
        #!Adding some randomness
        #param_pos_randomness=torch.rand(param_pos.shape).to(param_pos.device)*0.2-0.1
        #param_size_randomness=torch.rand(param_size.shape).to(param_size.device)*0.2-0.1
        #param_pos+=param_pos_randomness
        #param_size+=param_size_randomness
        
        #!Adding a better randomness
        if smooth:
            param_pos_randomness=(torch.rand(param_pos.shape).to(param_pos.device)*2-1)*param_pos_interval
            param_pos+=param_pos_randomness
            r=torch.rand(param_pos.shape[:-1]).unsqueeze(-1).to(param_pos.device)
            param_size=param_size_range[:,:,0:1]*r+param_size_range[:,:,1:2]*(1-r)
            param_size=torch.tile(param_size,[1,1,2])
        
        #!Post process to avoid black margin
        
        if avoid_black_margin:
            max_param_size=torch.minimum(torch.minimum(param_pos[...,0]+1, 1-param_pos[...,0]), torch.minimum(param_pos[...,1]+1, 1-param_pos[...,1]))
            max_param_size=max_param_size.unsqueeze(-1).tile([1,1,2])
            param_size=torch.minimum(param_size, max_param_size)
        
        #self.param=torch.cat([params_pos, params_pos*0+0.75], axis=-1) #!Only for one layer
        self.param=torch.cat([param_pos, param_size], axis=-1)
        
        #Old entropy term
        self.entropy_every=torch.unsqueeze(-(prob*logprob).sum(axis=-1), 1)#!
        #New entropy to avoid encouraging too many small patches
        
        
        samples_onehot=torch.nn.functional.one_hot(samples, logprob.shape[-1])
        self.sample_logprob=(samples_onehot*logprob.unsqueeze(1)).sum(axis=-1)
        
    @property
    def params(self):
        return {"width": self.width.detach().cpu(), "entropy": self.entropy_every.detach().cpu()}
    
    def rsample(self):
        return self.param, self.sample_logprob    #!only support one sample now
    
               
        
class Augmentation_ConvFeature_crop_only(nn.Module):
    """old function, replace by Augmentation_ConvFeature in augmentation_ConvFeature.py"""

    def __init__(self, conv, transform, cfg={}):
        super(Augmentation_ConvFeature, self).__init__()
        self.transform = transform                
        self.get_param = AmortizedParamDist_ConvFeature(conv)        
        self.distC=Cropping_Uniform_Dist_ConvFeature
                
    def parameters(self):
        return self.get_param.parameters()

    def forward(self, x):
        #return x, x[:,0,0,0]*0#!
        bs, _, w, h = x.size()
        
        ## Learnable but globla cropping, only for baseline
        if False:
            print('--------------uniform aug mode--------------------------')
            param, center, size=self.get_param(x*0)
        else:
            param, center, size=self.get_param(x)
        
        
        self.dist = self.distC(param, center, size)

        weights, sample_logprob = self.dist.rsample()
        
        weights, sample_logprob=weights[:,0,:], sample_logprob[:,0]#!Only supports single sample!
        
        ## Use fixed cropping width
        if False: #!
            #print('--------------fixed cropping width--------------------------')
            weights[...,2:]=weights[...,2:]*0+0.7

        ## Random cropping, only for baseline
        if False: #!
            #print('--------------random aug mode--------------------------')
            crop=T.RandomResizedCrop(64, scale=(0.7,1), ratio=(1.0,1.0))#!
            x_out=[]
            for i in range(x.shape[0]):
                x_out.append(crop(x[i]))
            x_out = torch.stack(x_out, dim=0)
            return x_out, self.dist.param[:,:2]
            
        ## No cropping
        if False: #!
            #print('--------------no aug mode--------------------------')
            return x, self.dist.param[:,:2]
                
        ## exponential map
        i=0
        for tranformation in self.transform:
            weights_dim=Generators.weights_dim(tranformation)
            mat = getattr(Generators, tranformation)(weights[:,i:i+weights_dim])
            if i==0:
                affine_matrices=mat
            else:
                affine_matrices=torch.matmul(affine_matrices, mat)
            i+=weights_dim
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(), align_corners=True)
        x_out = F.grid_sample(x, flowgrid, align_corners=True)
        
        self.entropy_every=self.dist.entropy_every
        
        return x_out, sample_logprob #!Only for old model
        #!return x_out, self.dist.param[:,:2]        
        
        
if __name__=='__main__':
    pass