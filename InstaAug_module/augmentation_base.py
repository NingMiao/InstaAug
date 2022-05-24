import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import expm, MoG_entropy
import torchvision.transforms as T


from torch import distributions
import numpy as np

#!torch.autograd.set_detect_anomaly(True)

class GlobalParam(nn.Module):
    """Parameters are global, i.e. not data dependant

    Args:
        shape_param (tuple): Shape of the parameters to be output
    """

    def __init__(self, shape_param, init_param=None, **kwargs):
        super(GlobalParam, self).__init__()
        #!init_param = -2 * torch.ones(*shape_param) if init_param is None else torch.tensor(init_param)
        #!init_param =  torch.zeros(*shape_param) if init_param is None else torch.tensor(init_param)
        init_param =  torch.zeros(*shape_param)
        self._param = nn.Parameter(init_param)

    def parameters(self):
        return self._param

    def forward(self, x, **kwargs):
        #!return F.softplus(self._param).expand(x.size(0), *self._param.shape)
        
        return self._param.expand(x.size(0), *self._param.shape)


class FeaturizedParam(nn.Module):
    """Parameters are a linear transformation of the featurizer output

    Args:
        classifier (torch.nn.module): Neural network with a featurizer attribute outputing learned features
        shape_param (tuple): Shape of the parameters to be output
    """

    def __init__(self, classifier, shape_param, **kwargs):
        super(FeaturizedParam, self).__init__()
        self.featurizer = classifier.featurizer
        self.shape_param = shape_param
        self._param = nn.Linear(self.featurizer.output_dim, np.prod(shape_param))

    def parameters(self):
        return self._param.parameters()

    def forward(self, x):
        feature = self.featurizer(x)
        param = self._param(feature).view(-1, *self.shape_param)
        return F.softplus(param)


class AmortizedParam(nn.Module):
    """Parameters are a learnt from an amortized netork

    Args:
        shape_param (tuple): Shape of the parameters to be output
    """

    def __init__(self, net, shape_param, **kwargs):
        super(AmortizedParam, self).__init__()
        self.shape_param = shape_param
        self.net = net(num_targets=np.prod(shape_param))

    def parameters(self):
        return self.net.parameters()

    def forward(self, x):
        #return torch.zeros([x.shape[0],np.prod(self.shape_param)]) #!
        param = self.net(x).view(-1, *self.shape_param)
        #return F.softplus(param)*0#!
        return param/10.0 #!Leave activation to distribution

class AmortizedParamDist(nn.Module):
    """Parameters are a learnt from an amortized netork

    Args:
        shape_param (tuple): Shape of the parameters to be output
    """

    def __init__(self, net, mlp, shape_param, feature_size, z_dim, classifier, share_feature_with_classier=False, **kwargs):#!
        super(AmortizedParamDist, self).__init__()
        self.shape_param = shape_param
        self.share_feature_with_classier=share_feature_with_classier
        if share_feature_with_classier and classifier is not None:
            self.net = lambda x: classifier(x, True)
            feature_size=classifier.features_dim
            #feature_size=8192
        else:
            self.net = net(num_targets=feature_size)
        self.z_dim=z_dim
        self.mlp = mlp(num_inputs=feature_size+z_dim, num_targets=np.prod(shape_param))
    
    def parameters(self):
        if self.share_feature_with_classier:
            return list(self.mlp.parameters())
        return list(self.net.parameters())+list(self.mlp.parameters())

    def forward(self, x, z_num=20, z=None):

        feature = self.net(x)
        if z is None:
            if True:#!
                z = torch.randn([1, z_num, self.z_dim]).type_as(x).repeat(x.shape[0],1,1)#!
            else:
                z = torch.randn([x.shape[0], z_num, self.z_dim]).type_as(x)
            
        
        z_feature=torch.cat([z, feature.unsqueeze(1).expand([-1, z_num, -1])], axis=-1)
        z_feature_2_dim=z_feature.view(x.shape[0]*z_num, -1)        
        params=self.mlp(z_feature_2_dim).view(x.shape[0], z_num, -1)
        
        #return params/3.0 #~Leave activation to distribution
        return params #~Leave activation to distribution
    

class ParametricDistribution(nn.Module):
    """ Distributions for sampling transformations parameters """

    def __init__(self, **kwargs):
        super(ParametricDistribution, self).__init__()

    @property
    def volume(self):
        return self.width.norm(dim=-1)

    def rsample(self, shape):
        return self.distribution.rsample(shape)


class Uniform(ParametricDistribution):
    """ U([-theta, theta]) """

    len_param = 1

    def __init__(self, params, **kwargs):
        super(Uniform, self).__init__()
        self.width = params.squeeze(-1)
        self.distribution = distributions.Uniform(low=-self.width, high=self.width)

    @property
    def params(self):
        return {"width": self.width.detach().cpu()}

class Cropping_Uniform(ParametricDistribution):
    """ U([-theta, theta]) """

    len_param = 1

    def __init__(self, params, **kwargs):
        super(Cropping_Uniform, self).__init__()
        
        params_pos=F.sigmoid(params[:,:2])
        params_pos=0.8*2*(params_pos-0.5)
        
        params_size=F.sigmoid(params[:,2:]-3.0)
        params_size=-0.8*params_size+1.0
        #params_size=torch.min(params_pos+1, 1- params_pos)#!
        
        self.param=torch.cat([params_pos, params_size], axis=1)
        
        #self.width = 1-self.param[:,2]*self.param[:,3] #! Ning: Can be improved
        self.width = (2-self.param[:,2]-self.param[:,3])/2.0
        self.distribution = distributions.Uniform(low=params[:,0]*0, high=params[:,0]*1)
        
        zeros=torch.zeros([params.shape[0]]).type_as(params)
        self.zero_params=torch.stack([zeros, zeros, zeros+1, zeros+1], axis=1)

    @property
    def params(self):
        return {"width": self.width.detach().cpu()}
    
    def rsample(self, shape):
        r=torch.rand(self.zero_params.shape[0])[:,None].type_as(self.param)
        return self.param*r+self.zero_params*(1-r)

    
class Cropping_Uniform_Dist(ParametricDistribution):
    """ U([-theta, theta]) """

    len_param = 1

    def __init__(self, params, **kwargs):
        super(Cropping_Uniform_Dist, self).__init__()
        
        params_pos=F.sigmoid(params[:,:,:2])
        params_pos=0.8*2*(params_pos-0.5)
                
        if False:#!
            params_size=torch.min(params_pos+1, 1-params_pos)#!
        else:
            params_size=F.sigmoid(params[:,:,2:]-3.0)
            params_size=-0.8*params_size+1.0
        
        params_size=torch.min(params_pos+1, 1-params_pos)#!
        
        #params_pos=params_pos*0
        #params_size=params_size*0+1.0
        
        self.param=torch.cat([params_pos, params_size], axis=-1)
        
        self.entropy_every=MoG_entropy(params_pos)
        self.width=torch.mean((2-params_size[:,:,0]-params_size[:,:,1])/2.0, dim=1)
        
        self.param=self.param[:,0,:]
        zeros=torch.zeros([self.param.shape[0]]).type_as(params)
        self.zero_params=torch.stack([zeros, zeros, zeros+1, zeros+1], axis=1)
        
        #!self.diversity_every=MoG_entropy(params_pos.transpose(0,1))
        self.diversity_every=MoG_entropy(params_pos.mean(dim=1).unsqueeze(0), beta=1)#!
        
    @property
    def params(self):
        return {"width": self.width.detach().cpu(), "entropy": self.entropy_every.detach().cpu()}
    
    def rsample(self, shape):
        r=torch.rand(self.zero_params.shape[0])[:,None].type_as(self.param)
        return self.param*r+self.zero_params*(1-r)    
        #!return self.param
    
    @property
    def entropy(self):
        return self.entropy_every.mean(dim=-1)
        
    @property
    def diversity(self):
        return self.diversity_every.mean(dim=-1)


class AsymmetricUniform(ParametricDistribution):
    """ U([low, high]) """

    len_param = 2

    def __init__(self, params, **kwargs):
        super(AsymmetricUniform, self).__init__()
        self.low, self.high = params[..., 0], params[..., 1]
        self.distribution = distributions.Uniform(low=-self.low, high=self.high)

    @property
    def width(self):
        return (self.high + self.low) / 2

    @property
    def params(self):
        return {"low": self.low.detach().cpu(), "high": self.high.detach().cpu()}


class Beta(ParametricDistribution):
    """ Beta(alpha, beta) scaled to have support on [-theta, theta] """

    len_param = 3

    def __init__(self, params, **kwargs):
        super(Beta, self).__init__()
        self.alpha, self.beta, self.width = params[..., 0], params[..., 1], params[..., 2]
        self.distribution = distributions.Beta(concentration1=self.alpha, concentration0=self.beta)

    @property
    def params(self):
        return {"alpha": self.alpha.detach().cpu(), "beta": self.beta.detach().cpu(), "width": self.width.detach().cpu()}

    def rsample(self, shape):
        return self.distribution.rsample(shape) * 2 * self.width - self.width
    
class Augmentation(nn.Module):
    """docstring for MLPAug """

    def __init__(self, classifier, distC, paramC, transform, init_param=None, cfg={}):
        super(Augmentation, self).__init__()
        self.transform = transform
        
        param_dim=0
        for tranformation in self.transform:
            weights_dim=Generators.weights_dim(tranformation)
            param_dim+=weights_dim
        self.nb_transform = len(self.transform)

        #shape_param = (self.nb_transform, self.distC.len_param)
        # init_param = [-5.0, -5.0, 1.0, -5.0, -5.0, -5.0]  # NOTE: This is the param init in the original code
        # init_param = None
        additional_dict={}
        if hasattr(cfg, 'feature_size') and hasattr(cfg, 'z_dim'):
            additional_dict['feature_size']=cfg.feature_size
            additional_dict['z_dim']=cfg.z_dim
        
        self.get_param = paramC(classifier=classifier, shape_param=[param_dim], init_param=init_param, **additional_dict)
        
        self.distC=distC
                
    def parameters(self):
        return self.get_param.parameters()

    def forward(self, x, feature=None):
        
        if feature is None:
            feature=x
        
        bs, _, w, h = x.size()
        
        ## Learnable but globla cropping, only for baseline
        if False:
            print('--------------uniform aug mode--------------------------')
            param=self.get_param(feature*0)
        else:
            param=self.get_param(feature)
        
        
        self.dist = self.distC(param)
        weights = self.dist.rsample(torch.Size([]))
        
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

        return x_out #!Only for old model
        #!return x_out, self.dist.param[:,:2]


class Generators:
    def __init__(self):
        super(LieAlgebraGenerators, self).__init__()
    
    @staticmethod
    def weights_dim(transformation):
        dim_dict={'tx':1,
                  'ty':1,
                  'r':1,
                  'zoom':2,
                  'crop':4}
        return dim_dict[transformation]
                  
    #Don't need g[:, 2, 2]=torch.ones(bs) in all generators.
    @staticmethod
    def tx(weights):
        bs = weights.shape[0]
        g = torch.zeros(bs, 3, 3).to(weights.device)
        g[:, 0, 0]=torch.ones(bs)
        g[:, 1, 1]=torch.ones(bs)
        g[:, 2, 2]=torch.ones(bs)
        g[:, 0, 2] = weights[:,0]
        return g
    

    @staticmethod
    def ty(weights):
        bs = weights.shape[0]
        g = torch.zeros(bs, 3, 3).to(weights.device)
        g[:, 0, 0]=torch.ones(bs)
        g[:, 1, 1]=torch.ones(bs)
        g[:, 2, 2]=torch.ones(bs)
        g[:, 1, 2] = weights[:,0]
        return g

    @staticmethod
    def r(weights):
        bs = weights.shape[0]
        g = torch.zeros(bs, 3, 3).to(weights.device)
        g[:, 2, 2]=torch.ones(bs)
        g[:, 0, 0] = torch.cos(weights[:,0])
        g[:, 0, 1] = -torch.sin(weights[:,0])
        g[:, 1, 0] = torch.sin(weights[:,0])
        g[:, 1, 1] = torch.cos(weights[:,0])
        return g
    
    @staticmethod
    def zoom(weights):
        bs = weights.shape[0]
        g = torch.zeros(bs, 3, 3).to(weights.device)
        g[:, 2, 2]=torch.ones(bs)
        g[:, 0, 0] = weights[:,0]
        g[:, 1, 1] = weights[:,1]
        return g
    
    @staticmethod
    def crop(weights):
        #First center 
        g1=Generators.tx(-weights[:,0:1])
        g2=Generators.ty(-weights[:,1:2])
        #Then zoom in 
        g0=Generators.zoom(weights[:,2:4])
        g=torch.matmul(torch.matmul(g1,g2),g0)
        return g
        


class AugAveragedModel(nn.Module):
    """Feature averaging over augmentations

    Args:
        model (nn.module): classifier
        aug (nn.module): augmentation
        train_copies (int): number of augmentations to average over at training time
        test_copies (int): number of augmentations to average over at test time
    """

    def __init__(self, model, aug, train_copies, test_copies):
        super().__init__()
        self.aug = aug
        self.model = model
        self.train_copies = train_copies
        self.test_copies = test_copies

    def feature_averaging(self, x, n_copies):
        bs = x.shape[0]
        aug_x = torch.cat([self.aug(x) for _ in range(n_copies)], dim=0)
        return sum(torch.split(F.log_softmax(self.model(aug_x), dim=-1), bs)) / n_copies

    def forward(self, x):
        if self.training:
            #return self.model(self.aug(x))
            return self.feature_averaging(x, self.train_copies)
            #return self.model(x)
        else:
            # Faster batched implementation
            # return (sum(F.log_softmax(self.model(self.aug(x)),dim=-1) for _ in range(self.ncopies))/self.ncopies)#.log()  # Note: Originally commented
            return self.feature_averaging(x, self.test_copies)

        
if __name__=='__main__':
    pass