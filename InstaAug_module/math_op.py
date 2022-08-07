import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import expm, MoG_entropy
import torchvision.transforms as T


from torch import distributions
import numpy as np

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
        
class Crop:
    def __init__(self, size, zoom_min, zoom_max, zoom_step, translation_min, translation_max, translation_step, device):
        self.size, self.zoom_min, self.zoom_max, self.zoom_step, self.translation_min, self.translation_max, self.translation_step, self.device = size, zoom_min, zoom_max, zoom_step, translation_min, translation_max, translation_step, device
        self.zoom_tensor, zoom_stride = self.initialize_zoom(size, zoom_min, zoom_max, zoom_step, device) #Shape=[zoom_step+1, size, size]
        self.translation_tensor, translation_stride = self.initialize_translation(size, translation_min, translation_max, translation_step, device) #Shape=[translation_step+1, size, size]
        self.stride = torch.tensor([translation_stride,translation_stride,zoom_stride,zoom_stride], dtype=torch.float32, device=device).unsqueeze(0)
        self.min = torch.tensor([translation_min,translation_min,zoom_min,zoom_min], dtype=torch.float32, device=device).unsqueeze(0)
    
    def initialize_zoom(self, size, zoom_min=0.4, zoom_max=1.1, zoom_step=20, device='cpu'):
        #Based on action on x-axis (dimension 1), right multiplication
        zoom_tensor = torch.eye(size, dtype=torch.float32).unsqueeze(0).unsqueeze(0).tile(dims=(zoom_step+1,1,1,1)) #Shape=[1,1,size,size]
        zoom_stride = (zoom_max-zoom_min)/zoom_step
        weights_x = torch.arange(zoom_step+1, dtype=torch.float32)*zoom_stride+zoom_min
        weights_y = torch.ones(zoom_step+1, dtype=torch.float32)
        weights = torch.stack([weights_x, weights_y], axis=1)
                
        transformation='zoom'
        mat = getattr(Generators, transformation)(weights)
        affine_matrices=mat
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=zoom_tensor.size(), align_corners=True)
        zoom_tensor = F.grid_sample(zoom_tensor, flowgrid, align_corners=True)    #Shape=[batch,1,size,size]      
        
        
        return zoom_tensor.squeeze(1).to(device), zoom_stride
    
    def initialize_translation(self, size, translation_min=-1, translation_max=1, translation_step=20, device='cpu'):
        #Based on action on x-axis (dimension 1), right multiplication
        translation_tensor =  torch.eye(size, dtype=torch.float32).unsqueeze(0).unsqueeze(0).tile(dims=(translation_step+1,1,1,1))
        translation_stride = (translation_max-translation_min)/translation_step
        weights = torch.arange(translation_step+1, dtype=torch.float32) * translation_stride + translation_min
        weights = weights.unsqueeze(1)
                
        transformation='ty'
        mat = getattr(Generators, transformation)(weights)
        affine_matrices=mat
        
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=translation_tensor.size(), align_corners=True)
        translation_tensor = F.grid_sample(translation_tensor, flowgrid, align_corners=True)   #Shape=[batch,1,size,size]     
        
        return translation_tensor.squeeze(1).to(device), translation_stride
    
    def weights_to_mats(self, weights):
        #Map weights to grid
        weights_grid=((weights - self.min)/self.stride).to(torch.int32)
        
        
        #On x-axis
        ind = weights_grid[:,0]
        x_translation_mat = torch.index_select(self.translation_tensor, 0, ind).unsqueeze(-1) #[batch, size, size, 1]
        ind = weights_grid[:,2]
        x_zoom_mat = torch.index_select(self.zoom_tensor, 0, ind).unsqueeze(1) #[batch, 1, size, size]
        x_mat = (x_translation_mat*x_zoom_mat).sum(2) #[batch, size, size]
        
        #On y-axis
        ind = weights_grid[:,1]
        y_translation_mat = torch.index_select(self.translation_tensor, 0, ind).unsqueeze(-1) #[batch, size, size, 1]
        
        ind = weights_grid[:,3]
        y_zoom_mat = torch.index_select(self.zoom_tensor, 0, ind).unsqueeze(1) #[batch, 1, size, size]
        y_mat = (y_translation_mat*y_zoom_mat).sum(2).transpose(1,2) #[batch, size, size]
        
        return x_mat, y_mat
    
    def __call__(self, x, weights):
        #Shape x: [batch, 3, size, size], weights: [batch, 4]
        x_mat, y_mat = self.weights_to_mats(weights)
        x_mat_unsqueeze=x_mat.unsqueeze(1).unsqueeze(1) #Shape=[batch, 1, 1, size, size]
        x_1 = (x.unsqueeze(4)*x_mat_unsqueeze).sum(3) #Shape=[batch, 3, size, size]
        
        y_mat_unsqueeze=y_mat.unsqueeze(1).unsqueeze(4)
        x_2 = (x_1.unsqueeze(2)*y_mat_unsqueeze).sum(3)
        
        return x_2
        
        
        
        
if __name__=='__main__':
    
    img=torch.randn([3, 32, 32])
    
    def method_old(weight_np):
        x=torch.tensor(img).unsqueeze(0)
        weights=torch.tensor(weight_np, dtype=torch.float32).unsqueeze(0)
        transformation='crop'
        mat = getattr(Generators, transformation)(weights)
        affine_matrices=mat
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(), align_corners=True)
        x_out = F.grid_sample(x, flowgrid, align_corners=True)
        x_out=x_out.cpu().numpy()[0].transpose(1,2,0)
        return x_out
    
    crop=Crop(32, 0.3, 1.0, 200, -1, 1, 200, 'cpu')
    def method_new(weight_np):
        weights=torch.tensor(weight_np, dtype=torch.float32).unsqueeze(0)
        x_out = crop(torch.tensor(img).unsqueeze(0), weights)
        return x_out.numpy()[0].transpose(1,2,0)
    
    import numpy as np
    tx=np.random.random()*2-1
    ty=np.random.random()*2-1
    zx=np.random.random()*0.7+0.3
    zy=np.random.random()*0.7+0.3
    output_old=method_old(np.array([tx,ty,zx,zy], dtype=np.float32))
    output_new=method_new(np.array([tx,ty,zx,zy], dtype=np.float32))
    print(np.abs(output_old-output_new).mean())