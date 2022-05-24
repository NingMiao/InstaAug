"""
Base Learner wrapper definitions for logging, training and evaluating models
"""

import torch
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from mixmo.utils import misc, logger, config
from mixmo.learners import model_wrapper

import sys
sys.path.insert(0, '../../')
from InstaAug_module import Loss_Scheduler

import numpy as np

LOGGER = logger.get_logger(__name__, level="INFO")

#li_flag=aug_param.li_flag

class AbstractLearner:
    """
    Base learner class that groups models, optimizers and loggers
    Performs the entire model building, training and evaluating process
    """
    def __init__(self, config_args, Li_configs, dloader, device, Li=None):
        self.config_args = config_args
        self.device = device
        self.dloader = dloader
        self._tb_logger = None
        
        self.Li_configs=Li_configs
        self.Li=Li
        self.li_flag=self.Li_configs['li_flag']
        
        self._create_model_wrapper(Li_configs=Li_configs, Li=Li)

        self._best_acc = 0
        self._best_loss= 1e5
        self._best_epoch = 0
        
        self._init_scheduler()
    
    def _init_scheduler(self):
        self.real_epoch=0
        self.test_loss_scheduler=Loss_Scheduler(self.config_args['max_tolerance'])
        
        #if 'crop' in aug_param.mode and 'meanfield' not in aug_param.mode:
        #    self.crop_entropy_scheduler=Scheduler(aug_param.crop_min_threshold,aug_param.crop_max_threshold)
        #if 'color' in aug_param.mode:
        #    self.color_entropy_schedulers=[]
        #    for i in range(len(aug_param.color_min_thresholds)):
        #        self.color_entropy_schedulers.append(Scheduler(aug_param.color_min_thresholds[i], aug_param.color_max_thresholds[i]))
            
    
    def _create_model_wrapper(self, Li_configs, Li):
        """
        Initialize the model along with other elements through a ModelWrapper
        """
        self.model_wrapper = model_wrapper.ModelWrapper(
            config=self.config_args["model_wrapper"],
            config_args=self.config_args,
            Li_configs=Li_configs, 
            Li=Li,
            device=self.device
        )
        self.model_wrapper.to_eval_mode()
        self.model_wrapper.print_summary(
            pixels_size=self.dloader.properties("pixels_size")
            )

    @property
    def tb_logger(self):
        """
        Get (or initialize) the Tensorboard SummaryWriter
        """
        if self._tb_logger is None:
            self._tb_logger = SummaryWriter(log_dir=self.config_args["training"]["output_folder"])
        return self._tb_logger

    def save_tb(self, logs_dict, epoch):
        """
        Write stats from logs_dict at epoch to the Tensoboard summary writer
        """
        for tag in logs_dict:
            data=logs_dict[tag]["value"]
            if type(data)!=type([]) and type(data)!=type(np.array([1])):
                self.tb_logger.add_scalar(tag, logs_dict[tag]["value"], epoch)
            else:
                for i in range(len(data)):
                    self.tb_logger.add_scalar(tag+str(i), logs_dict[tag]["value"][i], epoch)
        #if "test/diversity_accuracy_mean" not in logs_dict:
        #    self.tb_logger.add_scalar(
        #        "test/diversity_accuracy_mean",
        #        logs_dict["test/accuracy"]["value"], epoch
        #    )

    def load_checkpoint(self, checkpoint, include_optimizer=True, return_epoch=False):
        """
        Load checkpoint (and optimizer if included) to the wrapped model
        """
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.model_wrapper.network.load_state_dict(checkpoint[self.model_wrapper.name + "_state_dict"], strict=True)
        if self.Li_configs['load_li']:
            #ckpt=checkpoint[self.model_wrapper.name + "_aug_state_dict"]
            #ckpt['augmentation.get_param.conv.conv_feature0.weight']=ckpt['augmentation.get_param.conv.conv_feature3.weight']
            #ckpt['augmentation.get_param.conv.conv_feature0.bias']=ckpt['augmentation.get_param.conv.conv_feature3.bias']
            #ckpt['augmentation.get_param.conv.conv_feature1.weight']=ckpt['augmentation.get_param.conv.conv_feature_global.weight']
            #ckpt['augmentation.get_param.conv.conv_feature1.bias']=ckpt['augmentation.get_param.conv.conv_feature_global.bias']
            #del(ckpt['augmentation.get_param.conv.conv_feature2.weight'])
            #del(ckpt["augmentation.get_param.conv.conv_feature2.bias"])
            #del(ckpt['augmentation.get_param.conv.conv_feature3.weight'])
            #del(ckpt["augmentation.get_param.conv.conv_feature3.bias"])
            #del(ckpt['augmentation.get_param.conv.conv_feature_global.weight'])
            #del(ckpt["augmentation.get_param.conv.conv_feature_global.bias"])

            for key in checkpoint[self.model_wrapper.name + "_aug_state_dict"].keys():#！
                print(key, checkpoint[self.model_wrapper.name + "_aug_state_dict"][key].shape)#!
            self.model_wrapper.Li.load_state_dict(checkpoint[self.model_wrapper.name + "_aug_state_dict"], strict=True)
            pass

        if include_optimizer:
            if self.model_wrapper.optimizer is not None:
                self.model_wrapper.optimizer.load_state_dict(
                    checkpoint[self.model_wrapper.name + "_optimizer_state_dict"])
            else:
                assert self.model_wrapper.name + "_optimizer_state_dict" not in checkpoint
        if return_epoch:
            return checkpoint["epoch"]

    def save_checkpoint(self, epoch, save_path=None):
        """
        Save model (and optimizer) state dict
        """
        # get save_path
        if epoch is not None:
            dict_to_save = {"epoch": epoch}
            if save_path is None:
                save_path = misc.get_model_path(
                    self.config_args["training"]["output_folder"], epoch=epoch
                )
        else:
            assert save_path is not None

        # update dict to save
        dict_to_save[self.model_wrapper.name + "_state_dict"] = (
            self.model_wrapper.network.state_dict()
            if isinstance(self.model_wrapper.network, torch.nn.DataParallel)
            else self.model_wrapper.network.state_dict())
        if self.li_flag:
            dict_to_save[self.model_wrapper.name + "_aug_state_dict"] = (
                self.model_wrapper.Li.state_dict())
        if self.model_wrapper.optimizer is not None:
            dict_to_save[self.model_wrapper.name + "_optimizer_state_dict"] = self.model_wrapper.optimizer.state_dict()

        # final save
        torch.save(dict_to_save, save_path)

    def train_loop(self, epoch):
        raise NotImplementedError

    def train(self, epoch):
        """
        Train for one epoch
        """
        self.model_wrapper.to_train_mode(epoch=epoch)
        if self.real_epoch==0 and self.li_flag:
            self._aug_scheduler(begining=True)
            self.begining=False
        # Train over the entire epoch
        
        self.train_loop(epoch)
        
        # Eval on epoch end
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch}"},
            }
        )
        scores = self.model_wrapper.get_dict_to_scores(split="train")
        for s in scores:
            logs_dict[s] = scores[s]

        ## Val scores
        if self.dloader.val_loader is not None:
            val_scores = self.evaluate(
                inference_loader=self.dloader.val_loader,
                split="val")
            for val_score in val_scores:
                logs_dict[val_score] = val_scores[val_score]

        ## Test scores
        test_scores = self.evaluate(
            inference_loader=self.dloader.test_loader,
            split="test")
        for test_score in test_scores:
            logs_dict[test_score] = test_scores[test_score]
        ## Print metrics
        misc.print_dict(logs_dict)

        ## Check if best epoch
        is_best_epoch = False
        if "test/accuracy" in logs_dict:
            ens_acc = float(logs_dict["test/accuracy"]["value"])
            if ens_acc >= self._best_acc:
                self._best_acc = ens_acc
                self._best_epoch = epoch
                is_best_epoch = True
        else:
            current_loss = float(logs_dict["test/main"]["value"])
            if current_loss <= self._best_loss:
                self._best_loss = current_loss
                self._best_epoch = epoch
                is_best_epoch = True
            

        ## Save the model checkpoint
        ## and not config.cfg.DEBUG
        if is_best_epoch:
            logs_dict["general/checkpoint_saved"] = {"value": 1.0, "string": "1.0"}
            save_epoch = True
        else:
            logs_dict["general/checkpoint_saved"] = {"value": 0.0, "string": "0.0"}
            save_epoch = (epoch % config.cfg.SAVE_EVERY_X_EPOCH == 0)
        
        if self.li_flag:
            augmentation_lr=self.model_wrapper.optimizer.param_groups[1]['lr']
            logs_dict["general/augmentation_lr"] = {"value": augmentation_lr, "string": str(augmentation_lr)}
            
            #if 'crop' in aug_param.mode and 'meanfield' not in aug_param.mode:
            #    entropy_weight=aug_param.crop_entropy_weight
            #    entropy_weight_value=entropy_weight
            #    logs_dict["general/crop_entropy_weight"] = {"value": entropy_weight_value, "string": str(entropy_weight)}
            #if 'color' in aug_param.mode=='color':
            if True:
                entropy_weight=self.Li_configs['entropy_weights']
                logs_dict["general/entropy_weight"] = {"value": entropy_weight, "string": str(entropy_weight)}
            
        
        if save_epoch:
            
            self.save_checkpoint(epoch)
            #self.save_checkpoint('best')#!
            LOGGER.warning(f"Epoch: {epoch} was saved")
        
        ## CSV logging

        short_logs_dict = OrderedDict(
            {k: v for k, v in logs_dict.items()
             if any([regex in k for regex in [
                 "test/accuracy",
                 "train/accuracy",
                 "epoch",
                 "checkpoint_saved",
                 "test/entropy",
                 "train/ce0",
                 "train/main",
                 "test/main",
                 "test/ce0",
                 "test/entropy",
                 "general/classifier_lr",
                 "general/augmentation_lr",
                 "general/entropy_weight",
                 ]])
            })
        misc.csv_writter(
            path=misc.get_logs_path(self.config_args["training"]["output_folder"]),
            dic=short_logs_dict
        )
        # Tensorboard logging
        if not config.cfg.DEBUG:
            self.save_tb(logs_dict, epoch=epoch)

        # Perform end of step procedure like scheduler update
        self.model_wrapper.scheduler.step()
        
        #Lr decay scheduler
        if 'test/ce0' in logs_dict:
            decrease_lr_flag=self.test_loss_scheduler.step(-logs_dict['test/accuracy']['value'])#!
        else:
            decrease_lr_flag=self.test_loss_scheduler.step(logs_dict['test/main']['value'])
        if decrease_lr_flag:
            #This only works for new Li
            self.model_wrapper.optimizer.param_groups[0]['lr']*=0.1
        
        # Dynamically update width_weight and entropy_weight(Scheduler)
        if self.li_flag:
            self._aug_scheduler(logs_dict)
    
    def _aug_scheduler(self, logs_dict=None, begining=False):
        #!if logs_dict['epoch']['value']<=aug_param.scheduler_sleep_epoch:
        #!    print('Aug scheduler sleeping!')
        #!    return 0
        
        #Width
        #!width_step=self.width_scheduler.step(logs_dict["test/width"]["value"])
        #!aug_param.width_weight*=width_step
        #!if width_step!=1:
        #!    print('width_weight: {}'.format(aug_param.width_weight))
        
        #Entropy
        #!entropy_step=self.entropy_scheduler.step(logs_dict["test/entropy"]["value"])
        #!aug_param.entropy_weight*=entropy_step
        #!if entropy_step!=1:
        #!    print('entropy_weight: {}'.format(aug_param.entropy_weight))
        
        
        #Introduce Li module slowly after several epochs
        self.real_epoch+=1
        epoch=self.real_epoch
        if 'warmup_period' in self.Li_configs and self.Li_configs['warmup_period']>0 and epoch<self.Li_configs['warmup_period']:
            self.model_wrapper.optimizer.param_groups[1]['lr']=self.Li_configs['lr']/self.Li_configs['warmup_period']*epoch
        else:
            self.model_wrapper.optimizer.param_groups[1]['lr']=self.Li_configs['lr']
        if begining:
            return 0
        
        
        
        #if 'crop' in aug_param.mode and 'meanfield' not in aug_param.mode:
        #    #Crop entropy term
        #    crop_entropy_step=self.crop_entropy_scheduler.step(logs_dict["test/crop_entropy"]["value"])
        #    aug_param.crop_entropy_weight*=crop_entropy_step
        #    if crop_entropy_step!=1:
        #        print('crop entropy weight: {}'.format(aug_param.crop_entropy_weight))
        #if 'color' in aug_param.mode:
            #Color entropy term
        if True:
            for i in range(len(self.Li_configs['entropy_min_thresholds'])):
                entropy_step=self.Li.schedulers[i].step(logs_dict["test/entropy"]["value"][i])
                self.Li_configs['entropy_weights'][i]*=entropy_step
                if entropy_step!=1:
                    print('entropy_weight: {}, {}'.format(i, self.Li_configs['entropy_weights'][i]))
        
        
        

                                          
        
    def evaluate_loop(self, dloader, verbose, **kwargs):
        raise NotImplementedError

    def evaluate(self, inference_loader, split="test", train_data=False):
        """
        Perform an evaluation of the model
        """
        # Restart stats
        self.model_wrapper.to_eval_mode()

        # Evaluation over the dataset properly speaking
        self.evaluate_loop(inference_loader, train_data=train_data)

        # Gather scores
        scores = self.model_wrapper.get_dict_to_scores(split=split)

        return scores
