"""
Training and evaluation loop definitions for the Learner objects
"""

from tqdm import tqdm

import torch

from mixmo.utils import logger, config
from mixmo.learners import abstract_learner
import torch.nn.functional as F
import numpy as np

LOGGER = logger.get_logger(__name__, level="DEBUG")

class Learner(abstract_learner.AbstractLearner):
    """
    Learner object that defines the specific train and test loops for the model
    """
    def _subloop(self, dict_tensors, backprop, backprop_aug=False):#!
        """
        Basic subloop for a step/batch (without optimization)
        """
        # Format input
        input_model = {"pixels": dict_tensors["pixels"]}
        if "metadata" in dict_tensors:
            input_model["metadata"] = dict_tensors["metadata"]
                    
        # Forward pass
        output_network = self.model_wrapper.predict(
            input_model, aug=self.li_flag, y=dict_tensors["target"]["target_0"])

        # Compute loss, backward and metrics
        self.model_wrapper.step(
            output=output_network,
            target=dict_tensors["target"],
            backprop=backprop,
            backprop_aug=backprop_aug,
        )

        return self.model_wrapper.get_short_logs()
    
    def _train_subloop(self, dict_tensors,):
        """
        Complete training step for a batch, return summary logs
        """
        if not self.li_flag:
            # Reset optimizers
            self.model_wrapper.optimizer.zero_grad()
            # Backprop
            dict_to_log = self._subloop(dict_tensors, backprop=True, backprop_aug=False)
            # Optimizer step
            self.model_wrapper.optimizer.step()
        else:
            
            ## Update network
            group_network, group_aug = self.model_wrapper.optimizer.param_groups
            if group_network['lr']>0:
                # Reset optimizers
                self.model_wrapper.optimizer.zero_grad()
                # Backprop
                dict_to_log = self._subloop(dict_tensors, backprop=True, backprop_aug=False)
                # Optimizer step
                self.model_wrapper.optimizer.param_groups=[group_network]
            self.model_wrapper.optimizer.step()
            
            ## Update aug
            #Fix bugs about zero_grad()
            self.model_wrapper.network.eval() #This is important
            self.model_wrapper.optimizer.param_groups=[group_aug]
            self.model_wrapper.optimizer.zero_grad()
            # Backprop
            dict_to_log = self._subloop(dict_tensors, backprop=False, backprop_aug=True)
            # Optimizer step       
            self.model_wrapper.optimizer.step()
            self.model_wrapper.network.train()
            
            self.model_wrapper.optimizer.param_groups=[group_network, group_aug]
            
            
        return dict_to_log

    def train_loop(self, epoch):
        """
        Training loop for one epoch
        """
        # Set loop counter for the epoch
        #loop = tqdm(self.dloader.test_loader, dynamic_ncols=True)#!
        loop = tqdm(self.dloader.train_loader, dynamic_ncols=True)
        
        # Loop over all samples in the train set
        for batch_id, data in enumerate(loop):
            
            loop.set_description(f"Epoch {epoch}")

            # Prepare the batch
            #dict_tensors = self._prepare_batch_test(data)#!
            dict_tensors = self._prepare_batch_train(data)
                        
            # Perform the training step for the batch
            dict_to_log = self._train_subloop(dict_tensors=dict_tensors)
            del dict_tensors

            # Tie up end of step details
            loop.set_postfix(dict_to_log)
            loop.update()
            if config.cfg.DEBUG >= 2 and batch_id >= 10:
                break
            if self.model_wrapper.warmup_scheduler is not None:
                self.model_wrapper.warmup_scheduler.step()
            
            #batch_id
            #if batch_id>10:#!
            #    break#!
            #break#!
            
    def evaluate_loop(self, inference_loader, train_data=False):
        """
        Evaluation loop over the dataset specified by the loader
        """
        # Set loop counter for the loader/dataset
        loop = tqdm(inference_loader, disable=False, dynamic_ncols=True)
        
        
        # Loop over all samples in the evaluated dataset
        for batch_id, data in enumerate(loop):
            loop.set_description(f"Evaluation")

            # Prepare the batch
            if not train_data:
                dict_tensors = self._prepare_batch_test(data)
            else:
                dict_tensors = self._prepare_batch_train(data)

            # Forward over the batch, stats are logged internally
            with torch.no_grad():
                _ = self._subloop(dict_tensors, backprop=False)

            if config.cfg.DEBUG >= 2 and batch_id >= 10:
                break
                
            #batch_id
            #if batch_id>10:#!
            #    break#!
     

    def _evaluate_subloop_single(self, dict_tensors, backprop=False): #For analysis
        """
        Basic subloop for a step/batch (without optimization)
        """

        # Format input
        input_model = {"pixels": dict_tensors["pixels"]}
        input_model["metadata"] = {}
        if "metadata" in dict_tensors:
            input_model["metadata"] = dict_tensors["metadata"]

        # Forward pass
        logits = self.model_wrapper.predict(
            input_model)['logits']
        probs = F.softmax(logits, dim=1)        
        return probs
    
    def onehot(self, y, class_num=200): #For analysis
        y_onehot=np.zeros([y.shape[0], class_num], dtype=np.float32)
        for i in range(y.shape[0]):
            y_onehot[i, y[i]]=1
        return y_onehot
    
    def _evaluate_single(self, x, y): #For analysis
        dict_tensors = {"pixels": torch.tensor(x).to(self.device), 'meta_data':{}}
        probs = self._evaluate_subloop_single(dict_tensors)    
        probs_true=probs.detach().cpu().numpy()*self.onehot(y)
        probs_true=np.sum(probs_true, axis=1)
        return probs_true
    def evaluate_single(self, x, y, batch_size=100): #For analysis
        self.model_wrapper.to_eval_mode()
        l=int(np.ceil(x.shape[0]/batch_size))
        probs_true_list=[]
        for i in range(l):
            x_batch=x[i*batch_size:(i+1)*batch_size]
            y_batch=y[i*batch_size:(i+1)*batch_size]
            probs_true_batch=self._evaluate_single(x_batch, y_batch)
            probs_true_list.append(probs_true_batch)
        probs_true=np.concatenate(probs_true_list, axis=0)
        return probs_true    
    
    
    def train_single(self, epoch, x, y, steps=10):
        """
        Train for one epoch
        """
        #self.model_wrapper.to_train_mode(epoch=epoch)
        

        # Train over the entire epoch
        dict_tensors = {"pixels": torch.tensor(x).to(self.device),'target':{'target_0':torch.tensor(y).to(self.device)}, 'metadata':{}}
        for i in range(steps):
            self.model_wrapper.to_eval_mode()
            out=self._train_subloop(dict_tensors)

        ## Save the model checkpoint
        ## and not config.cfg.DEBUG

        self.save_checkpoint(epoch)
        LOGGER.warning(f"Epoch: {epoch} was saved")
        
        logs_dict={"test/entropy":{"value": self.Li.entropy()},
                  "epoch":{"value": epoch}}
        self._aug_scheduler(logs_dict)
        return out
        
    
    def _prepare_batch_train(self, data):
        """
        Prepares the train batch by setting up the input dictionary and putting tensors on devices
        """
        dict_tensors = {"pixels": [], "target": {}}
        
        if type(data['pixels_0'])==type([]):
            l=len(data['pixels_0'])
            for num_member in range(self.config_args["num_members"]):
                data["pixels_" + str(num_member)]=torch.cat(data["pixels_" + str(num_member)], dim=0)
                data['target_'+ str(num_member)]=torch.tile(data['target_'+ str(num_member)], [l,0])
        
        # Concatenate inputs along channel dimension and collect targets
        for num_member in range(self.config_args["num_members"]):
            dict_tensors["pixels"].append(data["pixels_" + str(num_member)])
            dict_tensors["target"]["target_" + str(num_member)] = data[
                "target_" + str(num_member)].to(self.device)
        dict_tensors["pixels"] = torch.cat(dict_tensors["pixels"], dim=1).to(self.device)

        # Pass along batch metadata
        dict_tensors["metadata"] = data.get("metadata", {})
        dict_tensors["metadata"]["mode"] = "train"

        return dict_tensors

    def _prepare_batch_test(self, data):
        """
        Prepares the test batch by setting up the input dictionary and putting tensors on devices
        """
        (pixels, target) = data
        if type(pixels)==type([]):
            l=len(pixels)
            pixels=torch.cat(pixels, dim=0)
            target=torch.tile(target, [l])
        dict_tensors = {
            "pixels": pixels.to(self.device),
            "target": {
                "target_" + str(num_member): target.to(self.device)
                for num_member in range(self.config_args["num_members"])
            },
            "metadata": {
                "mode": "inference"
            }
        }
        #np.save('tem/pixles.npy', pixels.detach().cpu().numpy())
                
        return dict_tensors
