"""
Utility definitions to wrap a model with losses, metrics and logs
"""

import copy
import torch
import torch.nn.functional as F
from collections import OrderedDict

from mixmo.networks import get_network
from mixmo.core import (
    loss, optimizer, temperature_scaling, scheduler,
    metrics_wrapper)
from mixmo.utils import logger, misc, torchsummary
from mixmo.utils.config import cfg


LOGGER = logger.get_logger(__name__, level="DEBUG")



def get_predictions(logits):
    """
    Convert logits into softmax predictions
    """
    probs = F.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1, keepdim=True)
    return confidence, pred, probs


class ModelWrapper:
    """Augment a model with losses, metrics, internal logs and other things
    """

    def __init__(self, config, config_args, Li_configs, Li, device):
        self.config = config
        self.name = config["name"]
        self.config_args = config_args
        self.Li_configs=Li_configs
        self.Li=Li
        self.li_flag=self.Li_configs['li_flag']
        
        self.device = device
        self.mode = "notinit"
        self._init_main()
        if self.li_flag:
            self.Li.to(self.device)

    def _init_main(self):
        self.network = get_network(
            config_network=self.config["network"], config_args=self.config_args
        ).to(self.device)
        self._scaled_network = None
        self.scheduler = None
        self._scheduler_initialized = False

        self.loss = loss.get_loss(
            config_loss=self.config.get("loss"),
            config_args=self.config_args, device=self.device,
            li_flag=self.li_flag
        )
        if hasattr(self.loss, "set_regularized_network"):
            self.loss.set_regularized_network(self.network)
        if not self.li_flag:
            self.optimizer = optimizer.get_optimizer(
            optimizer=self.config["optimizer"],
            list_param_groups=[{"params": list(self.network.parameters())}])
        else:
            self.optimizer = optimizer.get_optimizer(
            optimizer=self.config["optimizer"],
            list_param_groups=[{"params": list(self.network.parameters())}, {"params": list(self.Li.parameters()), 'lr':self.Li_configs['lr']}])
            
            
        #if hasattr(self.network, 'contrastive_pretraining'):
        #    from torchvision import transforms
        #    self.transform_crop=transforms.RandomCrop(64, padding=4) #!64 works tiny_imagenet only
        #    s=0.5
        #    self.transform_color=transforms.ColorJitter(0.8*s, 0.8*s, 0.0*s, 0.2*s)

    def to_eval_mode(self):
        """
        Switch model to eval mode
        """
        self.mode = "eval"
        self.network.eval()
        self.loss.start_accumulator()
        self._init_metrics()

    def to_train_mode(self, epoch):
        """
        Switch model to train mode
        """
        self.mode = "train"
        if not self._scheduler_initialized:
            self._init_scheduler(epoch)
        self.network.train()
        self.loss.start_accumulator()
        self._init_metrics()

    def _init_scheduler(self, epoch):

        self.scheduler = scheduler.get_scheduler(
            lr_schedule=self.config["lr_schedule"],
            optimizer=self.optimizer,
            start_epoch=epoch - 2,
        )
        self.scheduler.step()
        if epoch == 1 and self.config.get("warmup_period", 0) > 0:
            LOGGER.warning("Warmup period")
            self.warmup_scheduler = scheduler.get_warmup_scheduler(
                optimizer=self.optimizer,
                warmup_period=self.config.get("warmup_period"))
        else:
            self.warmup_scheduler = None
        self._scheduler_initialized = True

    def _init_metrics(self):
        if self.mode == "eval":
            metrics = [*self.config["metrics"]] + self.config.get("metrics_only_test", [])
        else:
            metrics = self.config["metrics"]
        self._metrics = metrics_wrapper.MetricsWrapper(metrics=metrics)

    def print_summary(self, pixels_size=32):
        summary_input = (3 * self.config_args["num_members"], pixels_size, pixels_size)
        try:
            torchsummary.summary(self.network, summary_input, list_dtype=None)
        except:
            LOGGER.warning("Torch summary failed", exc_info=True)

    def step(self, output, target=None, backprop=False, backprop_aug=False, retain_graph=True):
        """
        Compute loss, backward step and metrics if required by config
        Update internal records
        """
        if self.li_flag:
            entropy=self.Li.entropy(mean=False)
            current_loss, aug_loss = self.loss(output, target, entropy, self.Li_configs)

        else:
            current_loss, aug_loss = self.loss(output, target)
        if backprop:
            current_loss.backward(retain_graph=retain_graph)
        if backprop_aug:
            aug_loss.backward(retain_graph=retain_graph)
        
        if 'feature' in output:
            return None
        
        logits = output["logits" if self.mode != "train" else "logits_0"]
        
        target = target["target_0"]#!Bug: not correct for MIMO
        if logits.shape[0]==target.shape[0]*2:
            logits = logits[:int(logits.shape[0]/2)]
        confidence, pred, probs = get_predictions(logits)            
            
        if len(target.size()) == 2:
            target = target.argmax(axis=1)

        self._metrics.update(pred, target, confidence, probs)

        if self.mode != "train":
            self._compute_diversity(output, target)
    
    
    
    def _compute_diversity(self, output, target):
        """
        Compute diversity and update internal records
        """
        if self.config_args["num_members"] > 1:
            predictions = [
                output["logits_" + str(head)].max(dim=1, keepdim=False)[1].detach().to("cpu").numpy()
                for head in range(
                    0, self.config_args["num_members"])
            ]
            if self.config_args["num_members"] != 1:
                self._metrics.update_diversity(
                    target=[int(t) for t in target.detach().to("cpu").numpy()],
                    predictions=predictions,
                )

    def get_short_logs(self):
        """
        Return summary of internal records
        """
        return self.loss.get_accumulator_stats(format="short", split=None)

    def get_dict_to_scores(self, split,):
        """
        Format logs into a dictionary
        """
        logs_dict = OrderedDict({})
        if split == "train":
            lr_value = self.optimizer.param_groups[0]["lr"]
            logs_dict[f"general/{self.name}_lr"] = {
                "value": lr_value,
                "string": f"{lr_value:05.5}",
            }
        
        misc.clean_update(logs_dict, self.loss.get_accumulator_stats(format="long", split=split))
        
        if self.mode == "eval":
            LOGGER.info(f"Compute metrics for {self.name} at split: {split}")
            scores = self._metrics.get_scores(split=split)
            for s in scores:
                logs_dict[s] = scores[s]
        return logs_dict
 
    def predict(self, data, aug=False):
        """
        Perform a forward pass through the model and return the output
        """
        contrastive_pretraining=hasattr(self.scaled_network, 'contrastive_pretraining')
        
        if (not aug) or ((not self.Li_configs['test_time_aug']) and self.mode!='train'):#!
            #if not contrastive_pretraining:
            if True:
                output=self.scaled_network(data)
                output['logprob']=0#!
                try:
                    self.Li(data['pixels'])
                except:
                    pass
                
                if False:#!
                    file_name='tem/global_val_feature.npy'
                    import numpy as np
                    import os
                    primary_feature=self.network(data, True)[1].detach().cpu()
                    if os.path.exists(file_name):
                        old_data=np.load(file_name)
                        new_data=np.concatenate([old_data, primary_feature], axis=0)
                    else:
                        new_data=primary_feature
                    np.save(file_name, new_data)
                
                return output
            #else:
                #x=data['pixels']
                #shape=x.shape
                #shape_new=[-1, 3, shape[2], shape[3]]
                #x=x.reshape(shape_new)
                
                #x_multi=[]
                #for i in range(2):
                #    for j in range(x.shape[0]):
                #        x_transformed=self.transform_crop(x[i])
                #        x_transformed=self.transform_color(x_transformed)
                #        x_multi.append(x_transformed)
                #x_multi=torch.stack(x_multi, dim=0)
                #x_multi=torch.cat([x]*2, dim=0)
                #x_multi=self.transform_crop(x_multi)
                #x_multi=self.transform_color(x_multi)
                #data['pixels']=x_multi
                #if 'metadata' in data and 'mixmo_masks' in data['metadata']:
                #    data['metadata']["mixmo_masks"]=torch.tile(data['metadata']["mixmo_masks"],[2,1,1,1])
                #output = self.scaled_network(data)
                #return output
                
        
        #The following is the case with li
        x=data['pixels']
        shape=x.shape
        
        num_members=shape[1]//3
        shape_new=[-1, 3, shape[2], shape[3]]
        x=x.reshape(shape_new)
        
        if self.mode=='train':
            n_copies=self.Li_configs['train_copies']
        else:
            n_copies=self.Li_configs['test_copies']
        if contrastive_pretraining:
            n_copies=1#!Fixed to 2/1, may lead to bug

        if self.Li_configs['ConvFeature']:
            if self.mode=='train':
                aug_x, logprob=self.Li(x, n_copies=n_copies)
            else:
                aug_x, logprob=self.Li(x, n_copies=n_copies, output_max=n_copies)
        else:
            aug_x = torch.cat([self.Li(x) for _ in range(n_copies)], dim=0)#!bug and problem for MIMO        
        
        data['pixels']=aug_x.reshape([-1, shape[1], shape[2], shape[3]])
        
        if not self.Li_configs['ConvFeature']:        
            data['pixels']=torch.cat([data['pixels'], x], dim=0)

        if 'metadata' in data and 'mixmo_masks' in data['metadata']:
            data['metadata']["mixmo_masks"]=torch.tile(data['metadata']["mixmo_masks"],[n_copies+1,1,1,1])
        output = self.scaled_network(data)
        
        if False:#!#trial For trial only
            import numpy as np
            import os
            out_folder_name='../Learnable_invariance_package/crop_analysis/'
            #Need to clean the folder manually
            existing_files=[x for x in os.listdir(out_folder_name) if x.startswith('logit.')]
            if existing_files==[]:
                id_now=0
            else:
                id_before=np.max([int(x.split('.')[1]) for x in existing_files])
                id_now=id_before+1
            np.save(out_folder_name+'logit.'+str(id_now)+'.npy', output['logits'].detach().cpu().numpy())
            
        
        if contrastive_pretraining:#!
            output['logprob']=logprob#!
            return output#!
        
        if n_copies>1:
            for key in output:
                logit=output[key]
                
                
   
                if self.mode=='train':
                    logit_aug = sum(torch.split(F.log_softmax(logit[:n_copies*shape[0]], dim=-1), shape[0])) / n_copies
                    logprob = sum(torch.split(logprob, shape[0])) / n_copies
                else:
                    if key=='logits' and torch.sum(logprob)!=0: #For random or other method without logprob
                    #if False:
                        logit_new=F.log_softmax(logit[:n_copies*shape[0]])
                        logit_new=logit_new.reshape([n_copies, -1, logit_new.shape[-1]]).transpose(0,1)
                        logprob_new=logprob.reshape([n_copies, -1]).transpose(1,0).unsqueeze(-1)
                        logit_aug=torch.sum(torch.exp(logit_new)*torch.exp(logprob_new*0.0), dim=1)
                        
                    else:
                        #logit_aug = sum(torch.split(F.log_softmax(logit[:n_copies*shape[0]], dim=-1), shape[0]))
                        if False:
                            name='../Learnable_invariance_package/rawfoot_result/logit.npy'
                            import numpy as np
                            new=logit[:n_copies*shape[0]].detach().cpu().numpy()
                            try:
                                old=np.load(name)
                                save=np.concatenate([old, new], axis=0)
                            except:
                                save=new
                            np.save(name, save)
                        logit_aug = torch.log(sum(torch.split(torch.exp(5.0*F.log_softmax(logit[:n_copies*shape[0]], dim=-1)), shape[0]))/ n_copies)
                    logprob = torch.log(sum(torch.split(torch.exp(logprob), shape[0])) / n_copies)
                
                logit_x = logit[n_copies*shape[0]:]
                
                
                output[key]=torch.cat([logit_aug, logit_x], dim=0)
        
        ##Save for analysis
        if False:
            with open('../Learnable_invariance_package/rawfoot_result/logit.txt','a') as g:
                logit_np=logit.detach().cpu().numpy()
                s='\t'.join([' '.join([str(item) for item in line]) for line in logit_np])
                g.write(s+'\n')
        
        
        if self.Li_configs['ConvFeature']:
            output['logprob']=logprob
        
        return output
        

    @property
    def scaled_network(self):
        """
        Returns scaled_model if necessary for amp
        """
        if self._scaled_network is None:
            return self.network
        else:
            return self._scaled_network

    def calibrate_via_tempscale(self, tempscale_loader):
        """
        Returns calibrated temperature on val/test set
        """
        self.to_eval_mode()
        self._scaled_network = temperature_scaling.NetworkWithTemperature(
            network=self.network, device=self.device
        )
        self._scaled_network.learn_temperature_gridsearch(
            valid_loader=tempscale_loader,
            lrs=cfg.CALIBRATION.LRS,
            max_iters=cfg.CALIBRATION.MAX_ITERS
        )
        return self._scaled_network.temperature.cpu().detach().numpy()[0]
