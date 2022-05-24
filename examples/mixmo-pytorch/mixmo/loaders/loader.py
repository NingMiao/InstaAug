"""
Custom Dataloaders for each of the considered datasets
"""

import os

from torchvision import datasets

from mixmo.augmentations.standard_augmentations import get_default_composed_augmentations, GaussianBlur
from mixmo.loaders import cifar_dataset, abstract_loader
from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DatasetFromNumpy(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            #print(x)
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

class CIFAR10Loader(abstract_loader.AbstractDataLoader):
    """
    Loader for the CIFAR10 dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
        (self.augmentations_train, self.augmentations_test) = get_default_composed_augmentations(
            dataset_name="cifar", crop=self.crop, test_crop=self.test_crop, inset=self.inset, test_inset=self.test_inset
        )

    def _init_dataset(self, corruptions=False, **kwargs):
        self.train_dataset = cifar_dataset.CustomCIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        if not corruptions:
            self.test_dataset = cifar_dataset.CustomCIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.augmentations_test
            )
        else:
            self.test_dataset = cifar_dataset.CIFARCorruptions(
                root=self.corruptions_data_dir, train=False, transform=self.augmentations_test
            )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "cifar10-data")

    @property
    def corruptions_data_dir(self):
        return os.path.join(self.dataplace, "CIFAR-10-C")


    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (16, 32, 32),
            "conv1_is_half_size": False,
            "pixels_size": 32,
        }
        return dict_key_to_values[key]


class CIFAR100Loader(CIFAR10Loader):
    """
    Loader for the CIFAR100 dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataset(self, corruptions=False, **kwargs):
        self.train_dataset = cifar_dataset.CustomCIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        if not corruptions:
            self.test_dataset = cifar_dataset.CustomCIFAR100(
                root=self.data_dir, train=False, download=True, transform=self.augmentations_test
            )
        else:
            self.test_dataset = cifar_dataset.CIFARCorruptions(
                root=self.corruptions_data_dir, train=False, transform=self.augmentations_test
            )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "cifar100-data")

    @property
    def corruptions_data_dir(self):
        return os.path.join(self.dataplace, "CIFAR-100-C")

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class TinyImagenet200Loader(abstract_loader.AbstractDataLoader):
    """
    Loader for the TinyImageNet dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
        (self.augmentations_train, self.augmentations_test) = get_default_composed_augmentations("tinyimagenet", self.crop, self.test_crop, self.inset, self.test_inset, color=self.color, test_color=self.test_color, brightness=self.brightness, test_brightness=self.test_brightness)

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "tinyimagenet200-data")

    def _init_dataset(self, corruptions=False, brightness_select=False, **kwargs):
        if brightness_select:
            traindir = os.path.join(self.data_dir, 'train_light')#!new dataset
        else:
            traindir = os.path.join(self.data_dir, 'train')
        valdir = os.path.join(self.data_dir, 'val/images')#!
        #!valdir = os.path.join(self.data_dir, 'train')#!
        if self.n_views>1:
            from torchvision import transforms
            s=1.0
            size=64
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)            
            data_transforms = transforms.Compose([
                                              #transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),#!
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              #!GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
            self.augmentations_train=data_transforms
            self.augmentations_test=data_transforms
            
            
            self.train_dataset = datasets.ImageFolder(traindir, ContrastiveLearningViewGenerator(self.augmentations_train, n_views=self.n_views))
            self.test_dataset = datasets.ImageFolder(valdir, ContrastiveLearningViewGenerator(self.augmentations_test, n_views=self.n_views))
        else:
            self.train_dataset = datasets.ImageFolder(traindir,self.augmentations_train)
            self.test_dataset = datasets.ImageFolder(valdir,self.augmentations_test)
            
            
            if False: #!#trial Record labels
                import numpy as np
                label_list=[]
                for _,label in self.test_dataset:
                    label_list.append(int(label))
                label_array=np.array(label_list)
                out_folder_name='../Learnable_invariance_package/crop_analysis/'
                np.save(out_folder_name+'y'+'.npy', label_array)

    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (64, 32, 32),
            "conv1_is_half_size": True,
            "pixels_size": 64,
        }
        return dict_key_to_values[key]

class RawfootLoader(abstract_loader.AbstractDataLoader):
    """
    Loader for the TinyImageNet dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
               
        (self.augmentations_train, self.augmentations_test) = get_default_composed_augmentations("rawfoot", self.crop, self.test_crop, self.inset, self.test_inset, color=self.color, test_color=self.test_color, brightness=self.brightness, test_brightness=self.test_brightness, config_args=self.config_args)

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "rawfoot")

    def _init_dataset(self, corruptions=False, brightness_select=False, val_name='all', train_name='14'):
        traindir = os.path.join(self.data_dir, 'rawfoot-images-train-separated/'+train_name)        
        #if not os.path.exists(traindir):
        #    os.system('python dataset/process_rawfoot_images_average_train.py --folder dataset/rawfoot-images-train-separated --type_ids '+train_name)
        #    print(train_name)
        valdir = os.path.join(self.data_dir, 'rawfoot-images-val-separated/'+val_name)#!
        self.train_dataset = datasets.ImageFolder(traindir, self.augmentations_train)
        self.test_dataset = datasets.ImageFolder(valdir, self.augmentations_test)
        
        if False: #!#trial Record labels
            import numpy as np
            label_list=[]
            for _,label in self.test_dataset:
                label_list.append(int(label))
            label_array=np.array(label_list)
            out_folder_name='../Learnable_invariance_package/rawfoot_result/'
            np.save(out_folder_name+'y'+'.npy', label_array)

    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (64, 32, 32),
            "conv1_is_half_size": True,
            "pixels_size": 64,
        }
        return dict_key_to_values[key]
    
class MarioIggyLoader(abstract_loader.AbstractDataLoader):
    """
    Loader for the TinyImageNet dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
        self.augmentations_train= None
        self.augmentations_test=None

    @property
    def data_dir(self):
        return self.dataplace

    def _init_dataset(self, *args, **kwargs):
        from .mario_iggy import MarioIggy
        import numpy as np
        self.train_dataset= MarioIggy("mixmo/loaders/", train=True, max_rot_angle=np.pi / 2, n_train=1000, n_test=500, dataseed=88)        
        self.test_dataset= MarioIggy("mixmo/loaders/", train=False, max_rot_angle=np.pi / 2, n_train=1000, n_test=500, dataseed=89)
        #self.test_dataset=self.train_dataset#!


    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (64, 32, 32),
            "conv1_is_half_size": True,
            "pixels_size": 64,
        }
        return dict_key_to_values[key]    