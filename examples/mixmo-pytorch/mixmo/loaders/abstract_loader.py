"""
Base DataLoader definitions
"""
import os
import numpy as np
import torch

from mixmo.utils.logger import get_logger
from mixmo.loaders import dataset_wrapper, batch_repetition_sampler
from mixmo.utils.config import cfg

LOGGER = get_logger(__name__, level="DEBUG")


class AbstractDataLoader:
    """
    General dataloader that defines how loaders are built
    """

    def __init__(self, config_args, dataplace, split_test_val=False, corruptions=False, val_name=None, train_name=None):
        self.config_args = config_args
        self.dataplace = dataplace
        self.batch_size = int(config_args['training']['batch_size'])
        self.num_workers = 5 if not cfg.DEBUG else 0
        
        #;Be careful about the default value
        if 'crop' in config_args['data']: 
            self.crop=config_args['data']['crop']
        else:
            self.crop=True
        if 'test_crop' in config_args['data']: 
            self.test_crop=config_args['data']['test_crop']
        else:
            self.test_crop=True
        if 'inset' in config_args['data']: 
            self.inset=config_args['data']['inset']
        else:
            self.inset=True
        if 'test_inset' in config_args['data']: 
            self.test_inset=config_args['data']['test_inset']
        else:
            self.test_inset=True
        if 'color' in config_args['data']: 
            self.color=config_args['data']['color']
        else:
            self.color=True
        if 'test_color' in config_args['data']: 
            self.test_color=config_args['data']['test_color']
        else:
            self.test_color=True
        if 'brightness' in config_args['data']: 
            self.brightness=config_args['data']['brightness']
        else:
            self.brightness=True
        if 'test_brightness' in config_args['data']: 
            self.test_brightness=config_args['data']['test_brightness']
        else:
            self.test_brightness=True
        
        if 'n_views' in config_args['data']: 
            self.n_views=int(config_args['data']['n_views'])
        else:
            self.n_views=1
        
        
        self._init_dataaugmentations()
        if val_name:
            self._init_dataset(corruptions, val_name=val_name)
            if train_name:
                self._init_dataset(corruptions, val_name=val_name, train_name=train_name)
        else:
            self._init_dataset(corruptions)
            if train_name:
                self._init_dataset(corruptions, train_name=train_name)
        self._init_train_loader()
        self._init_valtest_loader(split_test_val)

    def _init_dataaugmentations(self):
        raise NotImplementedError

    def _init_dataset(self, corruptions=False):
        # self.train_dataset = None
        # self.test_dataset = None
        raise NotImplementedError

    def _init_train_loader(self):
        """
        Build the train loader with the proper sampler and data augmentations
        """
        # Choose the right dataset type
        if self.config_args["num_members"] > 1:
            class_dataset_wrapper = dataset_wrapper.MixMoDataset
        else:
            class_dataset_wrapper = dataset_wrapper.MSDADataset

        # Load augmentations
        self.traindatasetwrapper = class_dataset_wrapper(
            dataset=self.train_dataset,
            num_classes=int(self.config_args["data"]["num_classes"]),
            num_members=self.config_args["num_members"],
            dict_config=self.config_args["training"]["dataset_wrapper"],
            properties=self.properties
        )

        # Build standard sampler
        _train_sampler = torch.utils.data.sampler.RandomSampler(
            data_source=self.traindatasetwrapper,  ## only needed for its length
            num_samples=None,
            replacement=False,
        )

        # Wrap it with the repeating sampler used for multi-input models
        batch_sampler = batch_repetition_sampler.BatchRepetitionSampler(
            sampler=_train_sampler,
            batch_size=self.batch_size,
            num_members=self.config_args["num_members"],
            drop_last=True,
            config_batch_sampler=self.config_args["training"]["batch_sampler"]
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.traindatasetwrapper,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            batch_size=1,
            shuffle=False,
            sampler=None,
            drop_last=False,
            pin_memory=True,
        )
        
    def _init_valtest_loader(self, split_test_val):
        """
        Build the test (and possibly val) loader with the proper sampler and data augmentations
        """
        if not split_test_val:
            LOGGER.warning("No validation loader")
            self.val_loader = None
            self.test_loader = self.make_standard_loader(
                self.test_dataset, shuffle=False)
        else:
            split_ratio = 0.5
            LOGGER.warning("Validation size={split_ratio} taken from test".format(
                split_ratio=split_ratio))
            num_test = len(self.test_dataset)
            indices = list(range(num_test))

            test_idx_npy = os.path.join(self.data_dir, "test_idx.npy")
            val_idx_npy = os.path.join(self.data_dir, "val_idx.npy")
            if os.path.exists(test_idx_npy):
                LOGGER.warning("Loading existing test-val split indices")
                test_idx = np.load(test_idx_npy)
                val_idx = np.load(val_idx_npy)
            else:
                split = int(np.floor(split_ratio * num_test))
                np.random.seed(cfg.RANDOM.SEED_TESTVAL)
                np.random.shuffle(indices)
                val_idx, test_idx = indices[:split], indices[split:]
                np.save(test_idx_npy, test_idx)
                np.save(val_idx_npy, val_idx)

            # _init samplers
            test_dataset = torch.utils.data.Subset(self.test_dataset, test_idx)
            val_dataset = torch.utils.data.Subset(self.test_dataset, val_idx)

            # _init loaders
            self.val_loader = self.make_standard_loader(
                val_dataset, shuffle=False)
            self.test_loader = self.make_standard_loader(
                test_dataset, shuffle=False)

    def make_standard_loader(self, dataset, shuffle=True):
        """
        Build a dataloader from a dataset (wrapper on torch.utils)
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,#!
            drop_last=False,
            pin_memory=not (cfg.DEBUG > 0),
            num_workers=self.num_workers,
        )

    def make_corruptions_test_dataset(self):
        """
        Make robustness test dataset à la CIFAR10-C
        Prototype function (redefined for specific datasets)
        """
        corruptions_test_dataset = None
        return corruptions_test_dataset
