from .math_op import *
from .utils import RGB2HSV, HSV2RGB, normalize, denormalize
import time

class ParametricDistribution(nn.Module):
    """ Distributions for sampling transformations parameters """

    def __init__(self, **kwargs):
        super(ParametricDistribution, self).__init__()

    @property
    def volume(self):
        return self.width.norm(dim=-1)

    def rsample(self, shape):
        return self.distribution.rsample(shape)

#Only support cropping now!    
    
class Cropping_Categorical_Dist_ConvFeature(nn.Module):
    """ U([-theta, theta]) """
    #For our categorical cropping.
    len_param = 1

    def __init__(self, centers, center_intervals, sizes, size_ranges, device='cuda', **kwargs):
        super(Cropping_Categorical_Dist_ConvFeature, self).__init__()
        
        self.device=device
        center_list=[]
        size_list=[]
        
        center_interval_list=[]
        size_range_list=[]
        
        
        for i in range(len(centers)):
            center=centers[i]
            center_interval=center_intervals[i]
            size=sizes[i]
            size_range=size_ranges[i]
            
            center=center.reshape(-1, 2).to(device)
            center_list.append(center)
            
            zero_base=torch.zeros(center.shape).to(device)
            
            center_interval_list.append(zero_base+center_interval)
            
            size_list.append(zero_base+size)
            size_range_new=torch.stack([zero_base[:,0]+size_range[0], zero_base[:,1]+size_range[1]], dim=-1)
            size_range_list.append(size_range_new)
        
        
        self.centers=torch.cat(center_list, dim=0) #Shape:[num_patches, 2]
        self.center_intervals=torch.cat(center_interval_list, dim=0) #Shape:[num_patches, 2]
        self.sizes=torch.cat(size_list, dim=0) #Shape:[num_patches, 2]
        self.size_ranges=torch.cat(size_range_list, dim=0) #Shape:[num_patches, 2]        
        
        self.onehot_mat=torch.eye(self.centers.shape[0], device=device)
        
        self.output_memory=0
        
    @property
    def params(self):
        return {}
    
    def forward(self, params, n_copies=1, avoid_black_margin=False, smooth=True, output_max=0):   
        centers, center_intervals, sizes, size_ranges = self.centers, self.center_intervals, self.sizes, self.size_ranges
                
        
        param_list=[]
                
        for i in range(len(params)):
            param=params[i]            
            shape=param.shape
            param=param.view(shape[0], -1)
            param_list.append(param)
        
        
        params=torch.cat(param_list, dim=1) #Shape:[batch, num_patches]
        logprob=torch.nn.functional.log_softmax(params, dim=-1)        
        prob=torch.exp(logprob)        
        
        ##This is not supported by xla
        #samples=torch.multinomial(prob, n_copies, replacement=True)
        
        
        
        ##This is without replacement, careful
        rand = torch.empty_like(prob).uniform_()
        samples = (-rand.log()+logprob).topk(k=n_copies).indices #Careful log
        
        ##Select the top-output_max patches
        if output_max>0:#!May be incorrect
            smooth=False
            nums=output_max#!
            samples=torch.argsort(-logprob_single, axis=1)[:, :nums]                    
        #np.save('output/sample/samples_id_'+str(self.output_memory)+'.npy', samples.detach().cpu().numpy())#@
        self.output_memory+=1
        #np.save('output/centers.npy', centers.detach().cpu().numpy())#@
        #np.save('output/sizes.npy', sizes.detach().cpu().numpy())#@
        
        
        param_pos_list=[]
        param_pos_interval_list=[]
        param_size_list=[]
        param_size_range_list=[]
        for i in range(n_copies):
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
       
        #!Adding a better randomness
        if smooth:
            param_pos_randomness=(torch.rand(param_pos.shape).to(self.device)*2-1)*param_pos_interval
            param_pos+=param_pos_randomness
            r=torch.rand(param_pos.shape[:-1]).unsqueeze(-1).to(self.device)
            param_size=param_size_range[:,:,0:1]*r+param_size_range[:,:,1:2]*(1-r)
            param_size=torch.tile(param_size,[1,1,2])
        
        #!Post process to avoid black margin
        
        if avoid_black_margin:
            max_param_size=torch.minimum(torch.minimum(param_pos[...,0]+1, 1-param_pos[...,0]), torch.minimum(param_pos[...,1]+1, 1-param_pos[...,1]))
            max_param_size=max_param_size.unsqueeze(-1).tile([1,1,2])
            param_size=torch.minimum(param_size, max_param_size)
        
        transformation_param=torch.cat([param_pos, param_size], axis=-1)
        #Old entropy term
        entropy_every=torch.unsqueeze(-(prob*logprob).sum(axis=-1), 1)#!
        #New entropy to avoid encouraging too many small patches
        
        ##onehot not supported by xla
        #samples_onehot=torch.nn.functional.one_hot(samples, logprob.shape[-1])
        samples_onehot=torch.index_select(self.onehot_mat, 0, samples.reshape([-1])).reshape([samples.shape[0], samples.shape[1], -1])
        
        sample_logprob=(samples_onehot*logprob.unsqueeze(1)).sum(axis=-1)         
        
        #KL divergence which reflects the output difference between inputs
        prob_mean=prob.mean(axis=0, keepdim=True)
        log_prob_mean=torch.log(prob_mean)
        KL_every=(prob_mean*(log_prob_mean-logprob)).sum(axis=-1)
        
        return transformation_param, entropy_every, sample_logprob, KL_every

class Cropping_New_Param_Categorical_Dist_ConvFeature(nn.Module):
    """ U([-theta, theta]) """
    #For our categorical cropping.
    len_param = 1

    def __init__(self, device='cuda', **kwargs):
        super(Cropping_New_Param_Categorical_Dist_ConvFeature, self).__init__()
        
        self.device=device
        self.onehot_mat=torch.eye(238, device=device)
        
    @property
    def params(self):
        return {}
    
    def forward(self, logits, n_copies=1):   
       
        logprob=torch.nn.functional.log_softmax(logits, dim=-1)        
        prob=torch.exp(logprob)+1e-7        
        
        #import numpy
        #np.save('prob.npy', prob.detach().numpy())
        #print(prob)
        ##This is not supported by xla
        #samples=torch.multinomial(prob, n_copies, replacement=True)
        ##This is without replacement, careful
        rand = torch.empty_like(prob).uniform_()
        samples = (-rand.log()+logprob).topk(k=n_copies).indices #Careful log
        
        ##Select the top-output_max patches
        
        #Old entropy term
        #entropy_every=torch.unsqueeze(-(prob*logprob).sum(axis=-1), 1)#!
        
        #Split entropy into two terms
        prob_0=prob[:, :121]
        prob_1=prob[:, 121:221]
        prob_2=prob[:, 221:237]
        prob_3=prob[:, 237:]
        
        prob_list=[prob_0, prob_1, prob_2, prob_3]
        prob_sum_list=[torch.sum(prob_tem, dim=1, keepdims=True) for prob_tem in prob_list]
        prob_level=torch.cat(prob_sum_list, dim=1)
        
        #Old entropy
        #!entropy_every=torch.unsqueeze(-(prob_level*torch.log(prob_level)).sum(axis=-1), 1) 
        #Batch level entropy
        #prob_level_mean=torch.mean(prob_level, dim=0, keepdims=True)
        #entropy_every=-(prob_level_mean*torch.log(prob_level_mean)).sum(axis=-1, keepdims=True)
        #entropy_every=torch.tile(entropy_every, [prob_level.shape[0], 1])
        
        #size<=0.6 entropy
        entropy_every_pre0=prob[:,:121].sum(dim=1, keepdims=True)
        entropy_every_pre1=prob[:,121:221].sum(dim=1, keepdims=True)
        entropy_every=-torch.abs(0.3-entropy_every_pre0)-torch.abs(0.3-entropy_every_pre1)
        
        
        ##onehot not supported by xla
        #samples_onehot=torch.nn.functional.one_hot(samples, logprob.shape[-1])
        samples_onehot=torch.index_select(self.onehot_mat, 0, samples.reshape([-1])).reshape([samples.shape[0], samples.shape[1], -1])
        
        sample_logprob=(samples_onehot*logprob.unsqueeze(1)).sum(axis=-1)         
        
        #KL divergence which reflects the output difference between inputs
        prob_mean=prob.mean(axis=0, keepdim=True)
        log_prob_mean=torch.log(prob_mean)
        KL_every=(prob_mean*(log_prob_mean-logprob)).sum(axis=-1)
        
        return samples, entropy_every, sample_logprob, KL_every    
        
    
class Color_Uniform_Dist_ConvFeature(ParametricDistribution):
    """ U([-theta, theta]) """

    len_param = 1

    def __init__(self, param, transform_color=['h','s','v'], num=1, random_aug=False, **kwargs):
        super(Color_Uniform_Dist_ConvFeature, self).__init__()
                     
        if random_aug: #Fixed aug
            perturbation_range=torch.sigmoid(param*0+10)*1.0
            perturbation_range[:,:2]=perturbation_range[:,:2]*0
            perturbation_range[:,2:4]=perturbation_range[:,:2]*0+0.2
        else:
            perturbation_range=torch.sigmoid(param-4)#!For ours
            #!perturbation_range=torch.sigmoid(param)#!For global
        
        max_range=[]
        if 'h' in transform_color:
            max_range.extend([0.0, 0.0])#!
        if 's' in transform_color:
            max_range.extend([0.0, 0.0])#0.8 or 1.0?
        if 'v' in transform_color:
            max_range.extend([0.8, 0.8])#0.8 or 1.0?
        ranges=torch.tensor(max_range).type(perturbation_range.type()).reshape([1,-1,1,1])        
        perturbation_range=perturbation_range*ranges
        
        r_shape=list(perturbation_range.shape)
        r_shape[1]=int(r_shape[1]/2)
        r=torch.rand(r_shape).type(perturbation_range.type())
        perturbation=r*perturbation_range[:,::2]-(1-r)*perturbation_range[:,1::2]
        
        self.perturbation=perturbation
        
        #entropy_weight=torch.tensor(entropy_weights).type(perturbation_range.type()).reshape([1,6,1,1])
        #self.entropy_every=(perturbation_range*entropy_weight).sum(dim=1).mean(dim=[1, 2])
        self.entropy_every=perturbation_range.mean(dim=[2, 3])
        
        #import numpy as np
        #if np.random.random()<0.05:
        #    print(perturbation_range.mean(dim=[2, 3])[:3])
        
    @property
    def params(self):
        return {"entropy": self.entropy_every.detach().cpu()}
    
    def rsample(self):
        return self.perturbation    #!only support one sample now
    
class Rotation_Uniform_Dist_ConvFeature(ParametricDistribution):
    """ U([-theta, theta]) """

    len_param = 1

    def __init__(self, param, **kwargs):
        super(Rotation_Uniform_Dist_ConvFeature, self).__init__()
               
        import numpy as np
        perturbation_range=torch.sigmoid(param-2)*np.pi#!
        
        r_shape=list(perturbation_range.shape)
        r_shape[1]=int(r_shape[1]/2)
        r=torch.rand(r_shape).type(perturbation_range.type())
        perturbation=r*perturbation_range[:,::2]-(1-r)*perturbation_range[:,1::2]
        self.perturbation=perturbation
        
        #entropy_weight=torch.tensor(entropy_weights).type(perturbation_range.type()).reshape([1,6,1,1])
        #self.entropy_every=(perturbation_range*entropy_weight).sum(dim=1).mean(dim=[1, 2])
        self.entropy_every=perturbation_range.mean(dim=[2, 3])
        
        
        if np.random.random()<0.05:
            print(perturbation_range.mean(dim=[2, 3])[:3])
        
    @property
    def params(self):
        return {"entropy": self.entropy_every.detach().cpu()}
    
    def rsample(self):
        return self.perturbation    #!only support one sample now

class Crop_Meanfield_Uniform_Dist_ConvFeature(ParametricDistribution):
    """ U([-theta, theta]) """

    len_param = 1

    def __init__(self, param, **kwargs):
        super(Crop_Meanfield_Uniform_Dist_ConvFeature, self).__init__()
               
        import numpy as np
        perturbation_range_center=torch.sigmoid(param[:,:4]*0.1)*2-1#!
        perturbation_range_size=torch.sigmoid(param[:,4:]+4)#!
        perturbation_range=torch.cat([perturbation_range_center, perturbation_range_size], dim=1)
        
        r_shape=list(perturbation_range.shape)
        r_shape[1]=int(r_shape[1]/2)
        r=torch.rand(r_shape).type(perturbation_range.type())
        perturbation=r*perturbation_range[:,::2]+(1-r)*perturbation_range[:,1::2] #Different from other case
        self.perturbation=perturbation
        
        #entropy_weight=torch.tensor(entropy_weights).type(perturbation_range.type()).reshape([1,6,1,1])
        #self.entropy_every=(perturbation_range*entropy_weight).sum(dim=1).mean(dim=[1, 2])
        self.entropy_every=(perturbation_range[:,::2]-perturbation_range[:,1::2]).mean(dim=[2, 3])
        
        if np.random.random()<0.05:
            print(perturbation_range.mean(dim=[2, 3])[:3])
        
    @property
    def params(self):
        return {"entropy": self.entropy_every.detach().cpu()}
    
    def rsample(self):
        return self.perturbation    #!only support one sample now    
    
        
if __name__=='__main__':
    pass