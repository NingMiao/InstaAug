import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from copy import deepcopy as copy
import sys

#Utils
def softmax(logits):
    logits=logits-logits.min(axis=-1, keepdims=True)
    probs_un=np.exp(logits)
    probs=probs_un/np.sum(probs_un, axis=-1, keepdims=True)
    return probs
def save_figure(mat, path):
    fig = plt.figure(frameon=False)
    plt.imshow(mat, cmap='gray', vmin=0, vmax=2)
    #plt.xticks([0, 25, 50, 75, 99],['-180','-90','0','90','180'])
    fig.savefig(path, bbox_inches = 'tight', pad_inches = 0) 

def add_crop_rec(img, param, r=np.array([1.0,0.0,0.0]), zoom2=False):
    #Might have some error in values
    shape=img.shape
    if not zoom2:
        center_x=shape[1]/2*(1+param[0])
        center_y=shape[0]/2*(1-param[1])
        half_rec_x=shape[1]/2*param[2]
        half_rec_y=shape[0]/2*param[3]
    else:
        center_x=shape[1]/2*(1+0.5*param[0])
        center_y=shape[0]/2*(1-0.5*param[1])
        half_rec_x=shape[1]/4*param[2]
        half_rec_y=shape[0]/4*param[3]
 
    rec_x_max=max(min(int(center_x+half_rec_x), shape[1]-1),0)
    rec_x_min=max(min(int(center_x-half_rec_x), shape[1]-1),0)
    rec_y_max=max(min(int(center_y+half_rec_y), shape[0]-1),0)
    rec_y_min=max(min(int(center_y-half_rec_y), shape[0]-1),0)
    alpha=10
    for x in range(rec_x_min, rec_x_max+1):
        img[rec_y_max,x]=(alpha*r+img[rec_y_max,x])/(1+alpha)
        img[rec_y_min,x]=(alpha*r+img[rec_y_min,x])/(1+alpha)
    for y in range(rec_y_min, rec_y_max+1):
        img[y,rec_x_max]=(alpha*r+img[y,rec_x_max])/(1+alpha)
        img[y,rec_x_min]=(alpha*r+img[y,rec_x_min])/(1+alpha)
    return img

def zoom_out(img, mul=2):
    shape=img.shape
    img_new=np.zeros([int(shape[0]*mul), int(shape[1]*mul), shape[2]]).astype(img.dtype)
    img_new[int(shape[0]/2):int(shape[0]/2)+shape[0], int(shape[1]/2):int(shape[1]/2)+shape[1]]=img
    return img_new

dataset_wrapper={
    'mixmo':{
      'mix_method':{
        'method_name': 'null',
        'prob': 1,
        'replacement_method_name': 'mixup'},
      'alpha': 2,
      'weight_root': 3},
    'msda':{
      'beta': 1,
      'prob': 0.5,
      'mix_method': 'null'},
    'da_method': 'null'}
batch_sampler={
    'batch_repetitions': 1,
    'proba_input_repetition': 0}
       
        
class visualize:
    def __init__(self, Li, checkpoint, feature=True, inset=False, test_inset=False, ConvFeature=False):
        checkpoint = torch.load(checkpoint, map_location='cpu')
        Li.load_state_dict(checkpoint['classifier' + "_aug_state_dict"], strict=True)
        self.Li=Li
        self.ConvFeature=ConvFeature
        sys.path.insert(0,'../mixmo-pytorch')
        from mixmo.loaders import get_loader
        config_args={'data':{'dataset':'tinyimagenet200', 'num_classes': 200, 'crop': False, 'inset': inset, 'test_inset': test_inset }, 'training':{'batch_size':100, 'dataset_wrapper':dataset_wrapper, 'batch_sampler':batch_sampler}, 'num_members': 1}
        dloader = get_loader(config_args, dataplace='../mixmo-pytorch/dataset/') #!
        data_list=[]
        label_list=[]
        for data, label in dloader.test_loader:
            data_list.append(data)
            label_list.append(label)
        self.data=torch.cat(data_list, dim=0)
        self.label=torch.cat(label_list, dim=0)
        
        if feature:
            from mixmo.networks import get_network
            import pickle as pkl
            config1, config2=pkl.load(open('config_net.pkl','rb'))
            self.network = get_network(config_network=config1, config_args=config2)
            self.network.load_state_dict(checkpoint['classifier' + "_state_dict"], strict=True)
        else:
            self.network=None
            
    def __call__(self, num=10, seed=124, output_path='', zoom2=False, data=None):
        if data is None:
            inds=np.arange(self.data.shape[0])
            np.random.seed(seed)
            np.random.shuffle(inds)
            inds=inds[:num]        
            data=self.data[inds]
        
        if not self.ConvFeature:
            if self.network is None:
                params=self.Li.augmentation.get_param(data, 100) #!
            
                self.Li(data)#!
                print('diversity:', self.Li.diversity())#!
            else:
                _, feature=self.network(data, feature=True, feature_layer=2)
                #print(feature.shape)
                params=self.Li.augmentation.get_param(feature, 100)
                
            params_pos=F.sigmoid(params[:,:,:2])
            params_pos=0.8*2*(params_pos-0.5)        
            params_size=F.sigmoid(params[:,:,2:]-3.0)
            params_size=-0.8*params_size+1.0
        
            params_pos=params_pos.detach().cpu().numpy()
            params_size=params_size.detach().cpu().numpy()
            params=np.concatenate([params_pos, params_size], axis=-1)
        else:
            mats=[mat.detach().cpu().numpy() for mat in self.Li.augmentation.get_param(data)[0]]
            probs=[np.exp(x).sum() for x in mats]
            [print(x.shape) for x in mats]
            print(probs)
            mat=mats[1]
        
        imgs=data.detach().cpu().numpy()
        imgs=np.transpose(imgs, [0,2,3,1])
        
        f, axs = plt.subplots(len(imgs), 3, figsize=[12, 3*len(imgs)])
        if len(imgs)==1:
            axs=[axs]
        
        for i in range(len(imgs)):
            #original image
            axs[i][0].imshow(imgs[i])
            
            axs[i][0].set_xticks([])
            axs[i][0].set_yticks([])
            
            #Scatter point
            if not self.ConvFeature:
                axs[i][1].scatter(x=params_pos[i,:, 0], y=params_pos[i, :, 1], s=(params_size[i, :, 1]+params_size[i, :,0])**2*100, alpha=0.3) #？
                axs[i][1].set_xlim(-1, 1)
                axs[i][1].set_ylim(-1, 1)
                axs[i][1].set_xticks([])
                axs[i][1].set_yticks([])
            else:
                axs[i][1].imshow(mat[i])
            
            if not self.ConvFeature:
                #cropping examples
                if not zoom2:
                    img_crop=imgs[i]
                    np.random.seed(43)
                    for j in range(10):
                        color=np.random.uniform(0.2, 0.8, [3])
                        img_crop=add_crop_rec(img_crop, params[i,j], r=color)
                else:
                    img_crop=zoom_out(imgs[i],2)
                    np.random.seed(43)
                    for j in range(10):
                        color=np.random.uniform(0.2, 0.8, [3])
                        img_crop=add_crop_rec(img_crop, params[i,j], r=color, zoom2=True)
            
                axs[i][2].imshow(img_crop)
                axs[i][2].set_xticks([])
                axs[i][2].set_yticks([])
            
        name=output_path+'dist_crop.png'
        f.savefig(name, bbox_inches = 'tight', pad_inches = 0) 
    
    def crop(self, img, weights):
        from .augmentation import Generators
        img=torch.tensor(img)
        weights=torch.tensor(weights)
        affine_matrices = Generators.crop(weights)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=img.size(), align_corners=True)
        x_out = F.grid_sample(img, flowgrid, align_corners=True)
        return x_out.detach().cpu().numpy()
    
    def output_data(self, num=10, seed=124, crop=False):
        inds=np.arange(self.data.shape[0])
        np.random.seed(seed)
        np.random.shuffle(inds)
        inds=inds[:num]
        
        data=self.data[inds]
        label=self.label[inds]
        
        if crop:
            weights=[]
            sample_num=49
            inter=2/(sample_num+1)
            
            for i in range(sample_num):
                for j in range(sample_num):
                    x1_trans=-1+(i+0.5)*inter
                    x2_trans=-1+(j+0.5)*inter
                    x1_port=min(1-x1_trans, 1+x1_trans)*0+0.8#！
                    x2_port=min(1-x2_trans, 1+x2_trans)*0+0.8#！
                    weights.append([x1_trans, x2_trans, x1_port, x2_port])
            weights=np.array(weights)
            
            data_list=[]
            label_list=[]
            for i in range(num):
                data_single=np.tile(data[i:i+1], (int(sample_num**2),1,1,1))
                label_single=np.tile(label[i:i+1], (int(sample_num**2)))
                data_cropped=self.crop(data_single, weights)
                data_list.append(data_cropped)
                label_list.append(label_single)
            data=np.concatenate(data_list, axis=0)
            label=np.concatenate(label_list, axis=0)
        if crop:
            np.save('visualize_x.npy', data)
            np.save('visualize_y.npy', label)
        else:
            np.save('visualize_x_origin.npy', data)
            np.save('visualize_y_origin.npy', label)
   
    def analyze_heatmap(self, num=10, seed=124, learned=True, baseline=True, output_path=''):
        maps=np.load('visualize_result.npy')
        maps=np.reshape(maps,[-1,49,49])
        
        inds=np.arange(self.data.shape[0])
        np.random.seed(seed)
        np.random.shuffle(inds)
        inds=inds[:num]        
        data=self.data[inds]
        data_tensor=data
        
        imgs=data.detach().cpu().numpy()
        imgs=np.transpose(imgs, [0,2,3,1])
        
        
        if learned:
            #!Be careful about directions
            def get_density_matrix(params_pos):
                params_pos=copy(params_pos)
                sample_num=19
                inter=2/(sample_num+1)
                mat=np.zeros([sample_num,sample_num])
                params_pos=(params_pos+1)*sample_num/2
                params_pos=params_pos.astype(np.int32)
                for i in range(params_pos.shape[0]):
                    x1, x2 = -params_pos[i][1], params_pos[i][0]
                    mat[x1, x2]+=1
                return mat/np.max(mat)
            
            sample_size=10000
            
            if self.network is not None:
                _, feature=self.network(data, feature=True)
                params=self.Li.augmentation.get_param(feature, sample_size)
            else:
                params=self.Li.augmentation.get_param(data, sample_size) #!
                
            params_pos=F.sigmoid(params[:,:,:2])
            params_pos=0.8*2*(params_pos-0.5) 
            params_size=F.sigmoid(params[:,:,2:]-3.0)
            params_size=-0.8*params_size+1.0
        
            params_pos=params_pos.detach().cpu().numpy()
            params_size=params_size.detach().cpu().numpy()
            params=np.concatenate([params_pos, params_size], axis=-1)
            
            params=np.reshape(params, [-1, sample_size, 4])
            
            
        
        
        if learned and baseline:
            f, axs = plt.subplots(len(imgs), 3, figsize=[9, 3*len(imgs)])
        else:
            f, axs = plt.subplots(len(imgs), 2, figsize=[7, 3*len(imgs)])
        
        if len(imgs)==1:
            axs=[axs]
        
        for i in range(len(imgs)):
            #original image
            axs[i][0].imshow(imgs[i])
            
            axs[i][0].set_xticks([])
            axs[i][0].set_yticks([])
            
            #Real
            axis=1
            if baseline:
                axs[i][1].imshow(maps[i].T, vmin=0, vmax=1)
                axs[i][1].set_xticks([])
                axs[i][1].set_yticks([])
                axis+=1
            #Learned
            if learned:
                print(params[i,:,:2])
                mat=get_density_matrix(params[i,:,:2])
                #mat=mat**3#!
                axs[i][axis].imshow(mat)
                axs[i][axis].set_xticks([])
                axs[i][axis].set_yticks([])
                        
        name='heatmap.png'
        f.savefig(output_path+name, bbox_inches = 'tight', pad_inches = 0) 
    
if __name__=='__main__':
    from .__init__ import learnable_invariance, learnable_invariance_ConvFeature
    ConvFeature=True
    if ConvFeature:
        Li=learnable_invariance_ConvFeature()
    else:
        Li=learnable_invariance()
    checkpoint='../mixmo-pytorch/model/exp_tinyimagenet_res18_1net_standard_bar1_testConvFeature_cropping_layer_last_3_ew_0.1/checkpoint_epoch_011.ckpt'
    #checkpoint='../mixmo-pytorch/model/exp_tinyimagenet_res18_1net_standard_bar1_testtrain_single/checkpoint_epoch_008.ckpt'
    Vi=visualize(Li, checkpoint, test_inset=False, feature=False, ConvFeature=ConvFeature)
    Vi(zoom2=False, output_path=checkpoint[:-25])
    #Vi.output_data(crop=True)
    #Vi.analyze_heatmap(output_path='./')