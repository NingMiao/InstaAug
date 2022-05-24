"""
Basic augmentation procedures
"""

from torchvision import transforms
from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")
import numpy as np
import torch.nn as nn

class CustomCompose(transforms.Compose):

    def __call__(self, img, apply_postprocessing=True):
        for i, t in enumerate(self.transforms):

            if i == len(self.transforms) - 1 and not apply_postprocessing:
                return {"pixels": img, "postprocessing": t}

            img = t(img)
        return img

def random_inset(X1, X2):
    X2=X2[:,:,::2,::2]
    size=X1.shape[2]
    half_size=X1.shape[2]//2
    for i in range(X1.shape[0]):
        pos_list=np.random.randint(0, half_size, [2])
        #pos_list=np.random.randint(half_size//2, half_size//2+1, [2])
        X1[i,:,pos_list[0]:pos_list[0]+half_size,pos_list[1]:pos_list[1]+half_size]=X2[i]
    return X1

def random_inset_4(X1, X2):
    X2=X2[:,:,::4,::4]
    size=X1.shape[2]
    quater_size=X1.shape[2]//4
    pos_list_candidate=np.array([[0,0],[0,size-quater_size],[size-quater_size, 0],[size-quater_size, size-quater_size]])
    for i in range(X1.shape[0]):
        pos_list=pos_list_candidate[np.random.randint(0, 4, [])]
        X1[i,:,pos_list[0]:pos_list[0]+quater_size,pos_list[1]:pos_list[1]+quater_size]=X2[i]
    return X1

class Transpose():
    def __init__(self):
        pass
    def __call__(self, input_tensor):
        if len(input_tensor)==3:
            output_tensor=input_tensor.permute(0,2,1)
        else:
            output_tensor=input_tensor.permute(0,1,3,2)
        return output_tensor

class Inset():
    def __init__(self):
        pass
    def __call__(self, input_tensor):
        if len(input_tensor)==3:
            input_tensor=input_tensor.unsqueeze(0)
            len3=True
        else:
            len3=False
        output_tensor=random_inset(input_tensor*0, input_tensor)#!
        if len3==True:
            output_tensor=output_tensor.squeeze(0)
        return output_tensor

    
import torch    
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
    return torch.stack([H, S, L], dim=-1)
    
    
    
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

class Color_aug:
    def __init__(self, random=1.0):
        self.random=random
    def __call__(self, input_tensor):
        input_tensor=torch.permute(input_tensor, (1,2,0))
        input_tensor=RGB2HSV(input_tensor)
        if len(input_tensor.shape)==3:
            r=torch.rand([1,1,3])
        else:
            r=torch.rand([input_tensor.shape[0], 1,1,3])
        #input_tensor=input_tensor+(r-0.5)*2*self.random
        input_tensor[...,2]=input_tensor[...,2]*torch.exp((r[...,2]-0.5)*2*self.random) #!
        #input_tensor=input_tensor+(r-0.5)*2*self.random
        input_tensor=torch.minimum(input_tensor*0+1, input_tensor)
        input_tensor=torch.maximum(input_tensor*0, input_tensor)
        input_tensor=HSV2RGB(input_tensor)
        input_tensor=torch.permute(input_tensor, (2,0,1))
        return input_tensor
    
class Brightness_aug:
    def __init__(self, ranges=0.3):
        self.range=ranges
    def __call__(self, input_tensor):
        input_tensor=torch.permute(input_tensor, (1,2,0))
        
        input_tensor=RGB2HSV(input_tensor)
        
        r=torch.rand([])
        
        if r<0.5:
            input_tensor[...,2]=input_tensor[...,2]*0.3 #!
        else:
            input_tensor[...,2]=input_tensor[...,2]*3.3
        input_tensor=torch.minimum(input_tensor*0+1, input_tensor)
        input_tensor=torch.maximum(input_tensor*0, input_tensor)
        input_tensor=HSV2RGB(input_tensor)
        input_tensor=torch.permute(input_tensor, (2,0,1))
        return input_tensor    
    
        
def ColourDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.0*s, 0.2*s)
    #b, s, h
    #rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #rnd_gray = transforms.RandomGrayscale(p=0.2)
    #color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    #color_distort = transforms.Compose([rnd_color_jitter])
    color_distort = transforms.Compose([color_jitter])
    return color_distort    


class Deactivate_channel:
    def __init__(self, mode='rgb', channel=['r', 'g']):
        self.mode=mode
        self.channel=channel
    def __call__(self, input_tensor):
        input_tensor=torch.permute(input_tensor, (1,2,0))
        
        if self.mode=='hsv':
            input_tensor=RGB2HSV(input_tensor)
            if 'h' not in self.channel:
                input_tensor[...,0]=input_tensor[...,0]*0.0+0.5
            if 's' not in self.channel:
                input_tensor[...,1]=input_tensor[...,1]*0.0+0.5
            if 'v' not in self.channel:
                input_tensor[...,2]=input_tensor[...,2]*0.0+0.5
            input_tensor=torch.minimum(input_tensor*0+1, input_tensor)
            input_tensor=torch.maximum(input_tensor*0, input_tensor)
            input_tensor=HSV2RGB(input_tensor)
        else:
            if 'r' not in self.channel:
                input_tensor[...,0]=input_tensor[...,0]*0.0+0.5
            if 'g' not in self.channel:
                input_tensor[...,1]=input_tensor[...,1]*0.0+0.5
            if 'b' not in self.channel:
                input_tensor[...,2]=input_tensor[...,2]*0.0+0.5
        
        input_tensor=torch.permute(input_tensor, (2,0,1))
        return input_tensor

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img    
    
cifar_mean = (0.4913725490196078, 0.4823529411764706, 0.4466666666666667)
cifar_std = (0.2023, 0.1994, 0.2010)


def get_default_composed_augmentations(dataset_name, crop=False, test_crop=False, inset=False, test_inset=False, transpose=False, test_transpose=False, color=False, test_color=False, brightness=False, test_brightness=False, config_args={}):
    if dataset_name.startswith("cifar"):
        normalize = transforms.Normalize(cifar_mean, cifar_std)
        # Transformer for train set: random crops and horizontal flip
        transform_list=[CustomCompose([transforms.ToTensor()])]
        if transpose:
            transform_list.append(Transpose())
        if crop:
            transform_list.append(transforms.RandomCrop(32, padding=4))####Should use randomresizecrop
        transform_list.append(transforms.RandomHorizontalFlip())
        if inset:
            transform_list.append(Inset())
        transform_list.append(normalize)
        train_transformer = CustomCompose(transform_list)
        
        
        transform_list=[CustomCompose([transforms.ToTensor()])]
        if test_transpose:
            transform_list.append(Transpose())
        if test_crop:
            transform_list.append(transforms.RandomCrop(32, padding=4))####Should use randomresizecrop
        if test_inset:
            transform_list.append(Inset())
        
        transform_list.append(normalize)
        train_transformer = CustomCompose(transform_list)

    elif dataset_name.startswith("tinyimagenet") or dataset_name.startswith("rawfoot") or dataset_name.startswith("imagenet"):
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ##For train
        transform_list=[]
        if dataset_name.startswith("imagenet"):
            transform_list.append(transforms.RandomResizedCrop(224))
            transform_list.append(transforms.Resize([64, 64]))
        
        transform_list.append(CustomCompose([transforms.ToTensor()]))
        
        if False:
            transform_list.append(Deactivate_channel('hsv',['v','h']))
        
        if brightness:
            transform_list.append(Brightness_aug())
        
        if color or ('random_bounds' in config_args and config_args['random_bounds']!=[0,0,0]):
            #!transform_list.append(Color_aug())
            if config_args:
                if config_args['random_bounds']!=[0,0,0]:
                    h,s,v=config_args['random_bounds']
                    color_jitter = transforms.ColorJitter(v, s, 0.0, h)
                    transform_list.append(color_jitter)
                else:
                    transform_list.append(ColourDistortion(s=1.0))
            
        
        if crop:
            #!transform_list.append(transforms.RandomCrop(64, padding=4))
            #!transform_list.append(transforms.RandomResizedCrop(64, scale=(0.10,1), ratio=(1.0,1.0)))
            transform_list.append(transforms.RandomResizedCrop(64, scale=(0.08,1), ratio=(0.75,1.33)))
            
        if transpose:
            transform_list.append(Transpose())
        transform_list.append(transforms.RandomHorizontalFlip())
        if inset:
            transform_list.append(Inset())
        transform_list.append(normalize)
        train_transformer = CustomCompose(transform_list)
        
        #For test
        transform_list=[]
        
        if dataset_name.startswith("imagenet"):
            transform_list.append(transforms.Resize([73, 73]))
            transform_list.append(transforms.RandomResizedCrop(64))
            
        
        transform_list.append(CustomCompose([transforms.ToTensor()]))

        
        if test_brightness:
            transform_list.append(Brightness_aug())
        
        if test_color:
            transform_list.append(ColourDistortion(s=1.0))
        if test_crop:
            #transform_list.append(transforms.RandomCrop(64, padding=4))
            transform_list.append(transforms.RandomResizedCrop(64, scale=(0.8,1), ratio=(1.0,1.0)))
        
        if test_transpose:
            transform_list.append(Transpose())
        if test_inset:
            transform_list.append(Inset())
        transform_list.append(normalize)
        test_transformer = CustomCompose(transform_list)
        
    else:
        raise ValueError(dataset_name)

    return train_transformer, test_transformer
