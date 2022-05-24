from .augmentation_ConvFeature_base import *
from .utils import RGB2HSV, HSV2RGB, normalize, denormalize

class AmortizedParamDist_ConvFeature_Crop_and_Color(nn.Module):
    """Parameters are a learnt from an amortized netork

    Args:
        shape_param (tuple): Shape of the parameters to be output
    """

    def __init__(self, conv, crop_layer=[3,-1], bias=[1,3], color_layer=[], color_dim=[], rotation_layer=[], rotation_dim=[2], crop_meanfield_layer=[], crop_meanfield_dim=[6], *args, **kwargs):#! Default color_layer=[-1], color_dim=[3]
        super(AmortizedParamDist_ConvFeature_Crop_and_Color, self).__init__()
        layers=[]
        dims=[]
        crop_dim=[1 for layer in crop_layer]
        self.crop_layers_id=[]
        for i in range(len(crop_layer)):
            layer=crop_layer[i]
            dim=crop_dim[i]
            layers.append(layer)
            dims.append(dim)
            self.crop_layers_id.append(i)
        for i in range(len(color_layer)):
            layer=color_layer[i]
            dim=color_dim[i]
            if layer in layers:
                ids=layers.index(layer)
                dims[ids]+=dim
            else:
                layers.append(layer)
                dims.append(dim)
        for i in range(len(rotation_layer)):
            layer=rotation_layer[i]
            dim=rotation_dim[i]
            if layer in layers:
                ids=layers.index(layer)
                dims[ids]+=dim
            else:
                layers.append(layer)
                dims.append(dim)
        for i in range(len(crop_meanfield_layer)):
            layer=crop_meanfield_layer[i]
            dim=crop_meanfield_dim[i]
            if layer in layers:
                ids=layers.index(layer)
                dims[ids]+=dim
            else:
                layers.append(layer)
                dims.append(dim)
                
        self.conv=torch.nn.DataParallel(conv(output_layer=layers, output_dims=dims))
        #self.conv=conv(output_layer=layers, output_dims=dims)
        self.bias=bias #!
        
        self.layers=layers
        
        self.crop_layer=crop_layer
        self.color_layer=color_layer
        self.rotation_layer=rotation_layer
        self.crop_meanfield_layer=crop_meanfield_layer
        
        self.crop_dim=crop_dim
        self.color_dim=color_dim
        self.rotation_dim=rotation_dim
        self.crop_meanfield_dim=crop_meanfield_dim
        
        #Memory
        self.centers=None
        self.center_intervals=None
        self.scopes=None
        self.scope_ranges=None
        self.center_color=None
    
    def parameters(self):
        return self.conv.parameters()

    def forward(self, x, max_black_ratio=1.0):
        
        params=self.conv(x)
        #If only 'crop' or 'color' is activated, the other set of outputs are [].
        #For cropping
        if self.crop_layer!=[]:
            params_crop=[]
            for i in range(len(self.crop_layer)):
                layer=self.crop_layer[i]
                idx=self.layers.index(layer)
                params_crop.append(params[idx][:,:1])
            if not self.centers:
                self.centers, self.center_intervals=self.conv.module.center(interval=True)#!Careful
            centers, center_intervals=self.centers, self.center_intervals
            if not self.scopes:
                self.scopes, self.scope_ranges=self.conv.module.scope(ranges=True)#!Careful
            scopes, scope_ranges=self.scopes, self.scope_ranges
            centers_crop=[centers[i] for i in self.crop_layers_id]
            center_intervals_crop=[center_intervals[i] for i in self.crop_layers_id]
            scopes_crop=[scopes[i] for i in self.crop_layers_id]
            scope_ranges_crop=[scope_ranges[i] for i in self.crop_layers_id]
            if max_black_ratio<1:#!
                mask=self.conv.module.mask_for_no_padding(max_black_ratio)
                for i in range(len(centers_crop)):
                    m=mask[self.crop_layers_id[i]]
                    if m==0:
                        continue
                    centers_crop[i]=centers_crop[i][m:-m, m:-m]
                    params_crop[i]=params_crop[i][:,:,m:-m, m:-m]
            log_nums=[np.log(c.shape[0]*c.shape[0]) for c in centers_crop]
            params_crop=[params_crop[i]+self.bias[i]-log_nums[i] for i in range(len(params_crop))]#!
        else:
            params_crop, centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop=[],[],[],[],[]
        
        #For color aug
        if self.color_layer!=[]:
            layer=self.color_layer[0]
            dim=self.color_dim[0]
            idx=self.layers.index(layer)
            if layer in self.crop_layer:
                param_color=params[idx][:,1:1+dim]
            else:
                param_color=params[idx][:,:dim]
            if self.center_color==None:
                center_color=[self.conv.module.center(layer) for layer in self.color_layer][0]#?To be improved               
            center_color=self.center_color
        else:
            param_color=None
            center_color=None
            
        #For rotation aug
        if self.rotation_layer!=[]:
            layer=self.rotation_layer[0]
            idx=self.layers.index(layer)
            dim=self.rotation_dim[0]
            i=0
            if layer in self.crop_layer:
                i+=1
            if layer in self.color_layer:
                i+=self.color_dim[0]
            param_rotation=params[idx][:,i:i+dim]
        else:
            param_rotation=None
            
        #For crop_meanfield aug
        if self.crop_meanfield_layer!=[]:
            layer=self.crop_meanfield_layer[0]
            idx=self.layers.index(layer)
            dim=self.crop_meanfield_dim[0]
            i=0
            if layer in self.crop_layer:
                i+=1
            if layer in self.color_layer[0]:
                i+=self.color_dim[0]
            if layer in self.rotation_layer[0]:
                i+=self.rotation_dim[0]
            param_crop_meanfield=params[idx][:,i:i+dim]
        else:
            param_crop_meanfield=None
        
        
        return params_crop, centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop, param_color, center_color, param_rotation, param_crop_meanfield #~First three for cropping, next two for color aug    
    
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
        
        import numpy as np
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
    
    
class Augmentation_ConvFeature(nn.Module):
    """docstring for MLPAug """

    def __init__(self, conv, transform, cfg={}):
        super(Augmentation_ConvFeature, self).__init__()
        self.transform = transform   
        self.transform_crop=[t for t in transform if t in ['crop']]
        self.transform_color=[t for t in transform if t in ['h','s','v']]
        self.transform_rotation=[t for t in transform if t =='rotation']
        self.transform_crop_meanfield=[t for t in transform if t =='crop_meanfield']

        
        if len(self.transform_crop)>0:
            #The max entropy is {[3, -1]:4.15, [2, 3, -1]:5.77, [1,2,3,-1]:7.12}
            #crop_layer=[2, 3, -1]
            #bias=[0, 1, 3]
            #crop_layer=[3, -1]
            #bias=[1, 3]
            crop_layer=cfg['crop_layer']
            bias=cfg['crop_layer_bias']
        else:
            crop_layer=[]
            bias=[]
        
        if len(self.transform_color)>0:
            color_layer=[-1]
            color_dim=[len(self.transform_color)*2]
        else:
            color_layer=[]
            color_dim=[]
       
        if len(self.transform_rotation)>0:
            rotation_layer=[-1]
            rotation_dim=[len(self.transform_rotation)*2]
        else:
            rotation_layer=[]
            rotation_dim=[]
            
        if len(self.transform_crop_meanfield)>0:
            crop_meanfield_layer=[-1]
            crop_meanfield_dim=[len(self.transform_crop_meanfield)*6]
        else:
            crop_meanfield_layer=[]
            crop_meanfield_dim=[]
        
        
        self.get_param = AmortizedParamDist_ConvFeature_Crop_and_Color(conv, crop_layer=crop_layer, bias=bias, color_layer=color_layer, color_dim=color_dim, rotation_layer=rotation_layer, rotation_dim=rotation_dim, crop_meanfield_layer=crop_meanfield_layer, crop_meanfield_dim=crop_meanfield_dim) #!  color_layer changed to 3     
        self.distC_crop=Cropping_Uniform_Dist_ConvFeature
        self.distC_color=Color_Uniform_Dist_ConvFeature
        self.distC_rotation=Rotation_Uniform_Dist_ConvFeature
        self.distC_crop_meanfield=Crop_Meanfield_Uniform_Dist_ConvFeature
        #Memory
        self.color_weights=None
        if 'max_black_ratio' in cfg:
            self.max_black_ratio=cfg['max_black_ratio']
        else:
            self.max_black_ratio=1.0
         
        
    def parameters(self):
        return self.get_param.parameters()

    def forward(self, x, n_copies=1, output_max=0, global_aug=False, random_crop_aug=False, random_color_aug=False, hsv_input=True):
        
        #output_max=K is to output the top-K samples, which only works for cropping
        
        
        bs, _, w, h = x.size()
        
        
        if hsv_input:
            x_denormalize=denormalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #This step is time consuming
            #x_denormalize=x#!Be careful about denormalize
            x_hsv=RGB2HSV(x_denormalize.permute([0,2,3,1])) #!Be careful about denormalize
            x_hsv_transpose=x_hsv.permute([0,3,1,2])
            x_input=torch.cat([x, x_hsv_transpose], dim=1)
            x_hsv=torch.tile(x_hsv,[n_copies, 1,1,1])
        else:
            x_input=x

        
        ## Learnable but global cropping, only for baseline
        if global_aug:
            #print('--------------uniform aug mode--------------------------')
            params_crop, centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop, param_color, center_color, param_rotation, param_crop_meanfield=self.get_param(x_input*0, max_black_ratio=self.max_black_ratio)
        else:
            params_crop, centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop, param_color, center_color, param_rotation, param_crop_meanfield=self.get_param(x_input, max_black_ratio=self.max_black_ratio)
        x=torch.tile(x,[n_copies, 1,1,1])
        
        
        
        
        #Output for analysis
        if False: #!#trial
            import numpy as np
            import os
            
            out_folder_name='../Learnable_invariance_package/crop_analysis/contrastive/'
            #Need to clean the folder manually
            existing_files=[x for x in os.listdir(out_folder_name) if x.startswith('x.')]
            if existing_files==[]:
                id_now=0
            else:
                id_before=np.max([int(x.split('.')[1]) for x in existing_files])
                id_now=id_before+1
            
            np.save(out_folder_name+'x.'+str(id_now)+'.npy', x_denormalize.detach().cpu().numpy())
            np.save(out_folder_name+'others.npy', [x for x in [centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop]])
            
            if False: #Color
                np.save('../Learnable_invariance_package/rawfoot_sample_output/x_hsv.npy', x_hsv.detach().cpu().numpy())
                np.save('../Learnable_invariance_package/rawfoot_sample_output/param_color.npy', param_color.detach().cpu().numpy())
            
            if False: #Crop
                np.save(out_folder_name+'aug_param_2.'+str(id_now)+'.npy', params_crop[0].detach().cpu().numpy())
                np.save(out_folder_name+'aug_param_3.'+str(id_now)+'.npy', params_crop[1].detach().cpu().numpy())
                np.save(out_folder_name+'aug_param_n1.'+str(id_now)+'.npy', params_crop[2].detach().cpu().numpy())
                #np.save('../Learnable_invariance_package/aug_param_3.npy', params_crop[1].detach().cpu().numpy())
                #np.save('../Learnable_invariance_package/aug_param_n1.npy', params_crop[2].detach().cpu().numpy())
                
            if False: #Rotation
                np.save('../Learnable_invariance_package/param_rotation.npy', param_rotation.detach().cpu().numpy())
        
        #Color aug
        if len(self.transform_color)>0:
            param_color=torch.tile(param_color, [n_copies,1,1,1])#Haven't tested this yet
            self.dist_color=self.distC_color(param_color, transform_color=self.transform_color, random_aug=random_color_aug)
            perturbation=self.dist_color.rsample()
            if self.color_weights is None:
                n_pixel=w
                c_pixel=torch.zeros([n_pixel,n_pixel,2]).to(perturbation.device)
                base=2/n_pixel
                for i in range(n_pixel):
                    for j in range(n_pixel):
                        c_pixel[i,j,0]=-1+base/2+i*base
                        c_pixel[i,j,1]=-1+base/2+j*base
                c_pixel=c_pixel.view(-1, 1, 2).to(perturbation.device)
                c_center=center_color.reshape(1, -1, 2).to(perturbation.device)
                dis=((c_pixel-c_center)**2).sum(dim=-1)**0.5+1e-5
                dis_3=dis**(-3)
                weights=dis_3/dis_3.sum(dim=-1, keepdims=True)
                self.color_weights=weights
            perturbation=perturbation.view(perturbation.shape[0],perturbation.shape[1], -1)
            perturbation_pixel=torch.tensordot(perturbation, self.color_weights, dims=[[2],[1]])
            perturbation_pixel=perturbation_pixel.view(bs*n_copies,-1, w, h)
        
            x_hsv_list=[]
            perturbation_pixel=perturbation_pixel.permute([0,2,3,1])
            
            idx=0
            if 'h' in self.transform_color:
                idx=self.transform_color.index('h')
                x_hsv_new=perturbation_pixel[...,idx:idx+1]+x_hsv[...,0:1]
                x_hsv_list.append(x_hsv_new)
                idx+=1
            else:
                x_hsv_list.append(x_hsv[...,0:1])
        
            if 's' in self.transform_color:
                idx=self.transform_color.index('s')
                #x_hsv_new=torch.exp(perturbation_pixel[...,idx:idx+1])*x_hsv[...,1:2]
                x_hsv_new=(perturbation_pixel[...,idx:idx+1]+1)*x_hsv[...,1:2]
                x_hsv_list.append(x_hsv_new)
                idx+=1
            else:
                x_hsv_list.append(x_hsv[...,1:2])
        
            if 'v' in self.transform_color:
                idx=self.transform_color.index('v')
                #x_hsv_new=torch.exp(perturbation_pixel[...,idx:idx+1])*x_hsv[...,2:3]#! Not mul now
                x_hsv_new=(perturbation_pixel[...,idx:idx+1]+1)*x_hsv[...,2:3]
                x_hsv_list.append(x_hsv_new)
                idx+=1
            else:
                x_hsv_list.append(x_hsv[...,2:3])
            x_hsv=torch.cat(x_hsv_list, dim=3)
        
        
            x_hsv_new_12=torch.min(x_hsv[...,1:],torch.tensor(1).type(x.type()))
            x_hsv_new_12=torch.max(x_hsv_new_12,torch.tensor(0).type(x.type()))
            x_hsv_new_0=torch.remainder(x_hsv[...,:1], 1)
            x_hsv=torch.cat([x_hsv_new_0, x_hsv_new_12], dim=3)
            x_rgb=HSV2RGB(x_hsv).permute([0,3,1,2])
            x=x*0+normalize(x_rgb, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   #Adding x*0 makes it faster, it's very weird.     
        
        
        #Rotation aug
        if len(self.transform_rotation)>0:
            param_rotation=torch.tile(param_rotation, [n_copies,1,1,1])#Haven't tested this yet
            
            self.dist_rotation=self.distC_rotation(param_rotation)
            perturbation=self.dist_rotation.rsample()
            affine_matrices = getattr(Generators, 'r')(perturbation[...,0,0])     
            flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(), align_corners=True)
            x_out = F.grid_sample(x, flowgrid, align_corners=True)
            x=x_out
        
        #crop_meanfield aug
        if len(self.transform_crop_meanfield)>0:
            param_crop_meanfield=torch.tile(param_crop_meanfield, [n_copies,1,1,1])#Haven't tested this yet
            
            self.dist_crop_meanfield=self.distC_crop_meanfield(param_crop_meanfield)
            perturbation_pre=self.dist_crop_meanfield.rsample()[...,0,0]
            size=perturbation_pre[:,-1:]
            center=perturbation_pre[:,:-1]*(1-size)
            perturbation=torch.cat([center, size, size], dim=-1)
            
            
            affine_matrices = getattr(Generators, 'crop')(perturbation)     
            flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(), align_corners=True)
            x_out = F.grid_sample(x, flowgrid, align_corners=True)
            x_in=x #For save
            x=x_out
            if False:
                import numpy as np
                output_folder='../Learnable_invariance_package/tiny_meanfield_out/'
                np.save(output_folder+'x.npy', x_in.detach().cpu().numpy())
                np.save(output_folder+'permutation.npy', perturbation.detach().cpu().numpy())
                np.save(output_folder+'x_out.npy', x_out.detach().cpu().numpy())
                np.save(output_folder+'affine_matrices.npy', affine_matrices.detach().cpu().numpy())
                                    
        #Crop
        if 'crop' in self.transform:
            if random_crop_aug:
                self.dist_crop = self.distC_crop(params_crop, centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop, avoid_black_margin=False)#For code correctness
                #crop=T.RandomResizedCrop(x.shape[2], scale=(0.08,1), ratio=(0.75,1.33))#!
                crop=T.RandomResizedCrop(x.shape[2], scale=(0.2,1), ratio=(0.75,1.33), interpolation=3)
                x_out=[]
                for i in range(x.shape[0]):
                    x_out.append(crop(x[i]))
                x_out = torch.stack(x_out, dim=0)
                return x_out, torch.zeros([x.shape[0]]).type(x.type())
            
            
            params_crop=[torch.tile(param_crop, [n_copies,1,1,1]) for param_crop in params_crop]
            
            
            trial=False#!#trial
            if trial:
                self.dist_crop = self.distC_crop(params_crop, centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop, n_copies=n_copies, avoid_black_margin=False, smooth=False, trial=True)#!Trail model for further improving 
            else:
                self.dist_crop = self.distC_crop(params_crop, centers_crop, center_intervals_crop, scopes_crop, scope_ranges_crop, n_copies=n_copies, avoid_black_margin=False, output_max=output_max)#!avoid_black_margin is True for contrastive and False for supervised

            
            weights, sample_logprob = self.dist_crop.rsample()
            
            
            weights, sample_logprob=weights[:,0,:], sample_logprob[:,0]#!Only supports single sample!
            ## exponential map
            i=0
            for tranformation in self.transform_crop:
                weights_dim=Generators.weights_dim(tranformation)
                mat = getattr(Generators, tranformation)(weights[:,i:i+weights_dim])
                #print(weights[:,i:i+weights_dim])#!
                if i==0:
                    affine_matrices=mat
                else:
                    affine_matrices=torch.matmul(affine_matrices, mat)
                i+=weights_dim
            
            if trial:
                x=x[:bs].tile([321,1,1,1])
                
            flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(), align_corners=True)
            x_out = F.grid_sample(x, flowgrid, align_corners=True)
            
            if trial:
                np.save(out_folder_name+'logprob.'+str(id_now)+'.npy', sample_logprob.detach().cpu().numpy())

            #self.entropy_every=self.dist_crop.entropy_every+self.dist_color.entropy_every
            
            if False:#!
                import numpy as np
                np.save('analysis/tem/x.npy', x.detach().cpu().numpy())
                np.save('analysis/tem/x_out.npy', x_out.detach().cpu().numpy())
                print(haha)
            
            return x_out, sample_logprob  
        else:
            return x, torch.zeros([x.shape[0]]).type(x.type())
        
        
if __name__=='__main__':
    pass