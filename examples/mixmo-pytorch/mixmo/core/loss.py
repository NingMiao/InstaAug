"""
Base loss definitions
"""
from collections import OrderedDict
import copy
import torch
import torch.nn as nn

from mixmo.utils import misc, logger

import sys
sys.path.insert(0, '../../')
from InstaAug_module import aug_param

LOGGER = logger.get_logger(__name__, level="DEBUG")

def cross_entropy_transform(loss):
    start_point=1
    width=0.3
    max_step=3
    
    multiplier=torch.exp((loss>start_point).type(loss.type())*torch.min((loss-start_point)/width, torch.zeros(loss.shape).type(loss.type())+max_step)).detach()
    
    return loss*multiplier
    

class AbstractLoss(nn.modules.loss._Loss):
    """
    Base loss class defining printing and logging utilies
    """
    def __init__(self, config_args, device, config_loss=None):
        self.device = device
        self.config_args = config_args or {}
        self.config_loss = config_loss or {}
        self.name = self.config_loss["display_name"]
        nn.modules.loss._Loss.__init__(self)

    def print_details(self):
        LOGGER.info(f"Using loss: {self.config_loss} with name: {self.name}")

    def start_accumulator(self):
        self._accumulator_loss = 0
        self._accumulator_len = 0

    def get_accumulator_stats(self, format="short", split=None):
        """
        Gather tracked stats into a dictionary as formatted strings
        """
        if not self._accumulator_len:
            return {}

        stats = OrderedDict({})
        loss_value = self._accumulator_loss / self._accumulator_len

        if format == "long":
            assert split is not None
            key = split + "/" + self.name
            stats[key] = {
                "value": loss_value,
                "string": f"{loss_value:.5}",
            }
        else:
            # make it as short as possibe to fit on one line of tqdm postfix
            loss_string = f"{loss_value:.3}".replace("e-0", "-").replace("e-", "-")
            stats[self.name] = loss_string
        return stats

    def forward(self, input, target):
        current_loss = self._forward(input, target)
        self._accumulator_loss += current_loss.detach().to("cpu").numpy()
        self._accumulator_len += 1
        return current_loss

    def _forward(self, input, target):
        raise NotImplementedError

def softcrossentropyloss(input, target, mean=True ):      
    if len(target.size()) == 1:
        target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
        target = target.to(torch.float).to(input.device)
    logsoftmax = torch.nn.LogSoftmax(dim=1)
        
    if mean:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(-target * logsoftmax(input), dim=1)

class SoftCrossEntropyLoss(AbstractLoss):
    """
    Soft CrossEntropy loss that specifies the proper forward function for AbstractLoss
    """
    def _forward(self, input, target, mean=True):
        """
        Cross entropy that accepts soft targets
        Args:
            pred: predictions for neural network
            targets: targets, can be soft
            size_average: if false, sum is returned instead of mean

        Examples::

            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)

            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        return softcrossentropyloss(input, target, mean)

def piecewise_function(x, points=[[0,0], [1,1]], out_slope=[0.5, 0.3]):
    
    result=0
    
    result+=(x<points[0][0])*(points[0][1]+out_slope[0]*(x-points[0][0]))
                            
    for i in range(len(points)-1):
        tem=(points[i][1]*(points[i+1][0]-x))+points[i+1][1]*(x-points[i][0])/(points[i+1][0]-points[i][0])
        result+=(x>points[i][0])*(x<points[i+1][0])*tem
    
    result+=(x>points[-1][0])*(points[-1][1]+out_slope[1]*(x-points[-1][0]))  
    return result

def softaccloss(logits, target, mean=True):
    if len(target.size()) == 1:
        target = torch.nn.functional.one_hot(target, num_classes=logits.size(-1))
        target = target.to(torch.float).to(logits.device)
    
    logit_correct=torch.sum(target * logits, dim=1)
    logit_enermy_best=torch.max(logits-target*1000, dim=1).values.detach()
    logit_difference=logit_correct-logit_enermy_best.detach()
    logit_difference_transforms=piecewise_function(logit_difference, [[-0.2, -1],[0.2, 0]], [0.2, 0.0])
    
    if False:#!
        tem=logit_difference_transforms.detach().cpu().numpy()
        #print([[i, tem[i]] for i in range(tem.shape[0]) if tem[i]<0])
        print(tem[11])
    
    loss=-logit_difference_transforms
    
    if mean:
        return torch.mean(loss)
    else:
        return loss

class SoftAccLoss(AbstractLoss):
    """
    New loss that behave more similar to top-1 accuracy than cross entropy
    """
    def _forward(self, input, target, mean=True):

        return softaccloss(input, target, mean)

class CombinedLoss(AbstractLoss):
    """
    New loss that behave more similar to top-1 accuracy than cross entropy
    """
    def _forward(self, input, target, mean=True):

        loss_acc=softaccloss(input, target, mean=False)
        loss_ce=softcrossentropyloss(input, target, mean=False)
        gate=(loss_ce>3).float()
        loss=gate*loss_ce+(1-gate)*loss_acc
        if mean:
            return loss.mean()
        else:
            return loss
    
DICT_LOSS_STANDARD = {
    "soft_cross_entropy": SoftCrossEntropyLoss,
    #"soft_acc": CombinedLoss,
    "soft_acc": SoftAccLoss,
    "contrastive": SoftCrossEntropyLoss, #Temperatory
}


class WrapperLoss(AbstractLoss):
    """
    Wrapper around the multiple losses. Initialized from listloss.
    """
    def __init__(self, config_loss, config_args, device, li_flag=False):
        AbstractLoss.__init__(
            self,
            config_args=config_args,
            config_loss=config_loss,
            device=device
        )
        self.losses = self._init_get_losses()
        self.regularized_network = None
        self.li_flag=li_flag
        
    def _init_get_losses(self):
        """
        Initialize and gather losses from listloss
        """
        losses = []
        for ic, config_loss in enumerate(self.config_loss["listloss"]):
            if config_loss["coeff"] == "<num_members":
                config_loss["coeff"] = (1. if ic < self.config_args["num_members"] else 0)
            if config_loss["coeff"] == 0:
                LOGGER.debug(f"Skip loss: {config_loss}")
                continue

            loss_callable = get_loss(config_loss, device=self.device, config_args=self.config_args)
            loss = copy.deepcopy(config_loss)
            loss["callable"] = loss_callable
            losses.append(loss)
        return losses

    def print_details(self):
        return

    def start_accumulator(self):
        AbstractLoss.start_accumulator(self)
        for loss in self.losses:
            loss["callable"].start_accumulator()
        
        self._accumulator_entropy=0
        
    def _get_accumulator_stats(self, format="short", split=None):
        """
        Gather tracked stats into a dictionary as formatted strings
        """
        if not self._accumulator_len:
            return {}

        stats = OrderedDict({})
        loss_value = self._accumulator_loss / self._accumulator_len
        if self.li_flag:
            entropy_value = self._accumulator_entropy / self._accumulator_len
                        
        if format == "long":
            assert split is not None
            key = split + "/" + self.name
            stats[key] = {
                "value": loss_value,
                "string": f"{loss_value:.5}",
            }
            if self.li_flag:
                stats[split + "/" +'entropy'] = {
                        "value": entropy_value,
                        "string": str(entropy_value),
                    }

        else:
            # make it as short as possibe to fit on one line of tqdm postfix
            loss_string = f"l:{loss_value:.3} ".replace("e-0", "-").replace("e-", "-")
            
            if self.li_flag:
                try:
                    loss_string += f"ce:{entropy_value:.3} ".replace("e-0", "-").replace("e-", "-")
                except:
                    pass
            stats[self.name] = loss_string

        return stats
    
    def get_accumulator_stats(self, format="short", split=None):
        """
        Gather tracked stats into a dictionary as formatted strings
        """
        if not self._accumulator_len:
            return {}

        stats = self._get_accumulator_stats(format=format, split=split)

        if format == "long":
            # tensorboard logs
            if self.config_loss.get("l2_reg"):
                l2_reg = self.l2_reg().detach().to("cpu").numpy()
                stats["general/l2_reg"] = {
                    "value": l2_reg,
                    "string": f"{l2_reg:.4}",
                }
            for loss in self.losses:
                substats = loss["callable"].get_accumulator_stats(
                    format=format,
                    split=split,
                )
                misc.clean_update(stats, substats)

        return stats
    
    def forward(self, input, target, entropy=None, aug_param=None):
        current_loss, prediction_loss, aug_loss = self._forward(input, target, entropy, aug_param=aug_param)
        self._accumulator_loss += prediction_loss.detach().to("cpu").numpy()
        if self.li_flag:
            self._accumulator_entropy += entropy.detach().to("cpu").numpy().mean(axis=0)
        self._accumulator_len += 1
        return current_loss, aug_loss
    
    def _forward(self, input, target, entropy=None, aug_param=None):
        """
        Perform loss forwards for each sublosses and l2 reg
        """
        if not 'contrastive' in self.losses[0]['name']:
            computed_losses = [self._forward_subloss(loss, input, target) for loss in self.losses]
        else:
            computed_losses = [self._forward_subloss_contrastive_pretrain(loss, input) for loss in self.losses]
        stacked_computed_losses = torch.stack([x[0] for x in computed_losses])
        prediction_loss = stacked_computed_losses.sum()
        
        stacked_computed_losses_not_mean=torch.stack([x[1] for x in computed_losses], axis=1)[:,0]#Not suitable for MIMI        
                
        
        if self.config_loss.get("l2_reg"):
            final_loss = prediction_loss + self.l2_reg() * float(self.config_loss.get("l2_reg"))
        else:
            final_loss = prediction_loss
        
        if self.li_flag:
            #prediction_penalty=torch.exp((torch.maximum(prediction_loss-2.8, prediction_loss*0)/0.2))
            #prediction_penalty=prediction_penalty.detach() 
            
            if not aug_param['ConvFeature']:#Not workable branch
                l=int(stacked_computed_losses_not_mean.shape[0]/2)
                stacked_computed_losses_not_mean_transformed=cross_entropy_transform(stacked_computed_losses_not_mean[:l])
                
                gate_threshold=2.0#!Todo: change to soft version
                gate=(stacked_computed_losses_not_mean[l:]<gate_threshold).type(entropy.type())
                aug_loss_pre = stacked_computed_losses_not_mean_transformed   - entropy * float(aug_param['entropy_weight'])
                aug_loss=torch.sum(gate*aug_loss_pre)/(torch.sum(gate)+1e-5)
            
            else:
                #aug_loss=(stacked_computed_losses_not_mean*input['logprob']).mean()
                aug_loss=(stacked_computed_losses_not_mean.detach()*input['logprob']).mean()+stacked_computed_losses_not_mean.mean()#!
                aug_loss-=(entropy.mean(dim=0) * torch.tensor(aug_param['entropy_weights']).type(entropy.type())).sum()
        else:
            aug_loss=torch.tensor(0.0)
        
        return final_loss, prediction_loss, aug_loss

    def _forward_subloss(self, loss, input, target):
        """
        Standard loss forward for one of the sublosses
        """
        coeff = float(loss["coeff"])
        subloss_input = self._match_item(loss["input"], dict_tensors=input)
        
        subloss_target = self._match_item(loss["target"], dict_tensors=target)
                
        loss_mean = loss["callable"](input=subloss_input, target=subloss_target)
        loss_not_mean=loss["callable"]._forward(input=subloss_input, target=subloss_target, mean=False)
        
        #print(subloss_target.detach().cpu().numpy())#!
        #t='\n'.join([str(x) for x in subloss_target.detach().cpu().numpy()])
        #with open('test_target2.txt', 'a') as g:#!
        #    g.write(t+'\n')
        if False:#!
            file_name='tem/global_val_label.npy'
            import numpy as np
            import os
            if os.path.exists(file_name):
                old_data=np.load(file_name)
                new_data=np.concatenate([old_data, target['target_0'].detach().cpu().numpy()], axis=0)
            else:
                new_data=target['target_0'].detach().cpu().numpy()
            np.save(file_name, new_data)
        
        
        return loss_mean * coeff, loss_not_mean

    def _forward_subloss_contrastive_pretrain(self, loss, input, temperature=0.07):
        """
        Standard loss forward for one of the sublosses
        """
        coeff = float(loss["coeff"])
        feature = self._match_item(loss["input"], dict_tensors=input)
        
        feature_normalize=feature/((feature**2).sum(dim=-1)+1e-5).unsqueeze(-1)**0.5
        sim_matrix_0=torch.matmul(feature_normalize, feature_normalize.T)
        sim_matrix_1=sim_matrix_0-torch.eye(sim_matrix_0.shape[0]).type(sim_matrix_0.type())*100
        sim_matrix_2=torch.exp(sim_matrix_1/temperature)
        
        real_batch_size=int(sim_matrix_0.shape[0]/2)
        sub_eye_matrix=torch.eye(real_batch_size).type(sim_matrix_0.type())
        sub_zero_matrix=torch.zeros([real_batch_size, real_batch_size]).type(sim_matrix_0.type())
        gather_matrix=torch.concat([torch.concat([sub_zero_matrix, sub_eye_matrix], dim=1), torch.concat([sub_eye_matrix, sub_zero_matrix], dim=1)], dim=0)
        
        loss_not_mean= -(sim_matrix_1/temperature*gather_matrix).sum(dim=1)+torch.log(sim_matrix_2.sum(dim=1))
        loss_mean=loss_not_mean.mean()
        #print(sim_matrix_1.mean(), sim_matrix_2.mean(), feature.mean(), feature_normalize.mean(), loss_mean)
        #import numpy as np
        #np.save('tem/feature.npy', feature.detach().cpu().numpy())
        #np.save('tem/feature_normalize.npy', feature_normalize.detach().cpu().numpy())
        #np.save('tem/sim_matrix_1.npy', sim_matrix_1.detach().cpu().numpy())
        #np.save('tem/sim_matrix_2.npy', sim_matrix_2.detach().cpu().numpy())
        
        return loss_mean * coeff, loss_not_mean
    
    
    @staticmethod
    def _match_item(name, dict_tensors):
        if misc.is_none(name):
            return None
        if name in dict_tensors:
            return dict_tensors[str(name)]
        raise ValueError(name)

    def set_regularized_network(self, network):
        if self.config_loss.get("l2_reg"):
            self.regularized_network = network
            LOGGER.warning(f"Set l2 regularization on {network.__class__.__name__}")

    def l2_reg(self,):
        """
        Compute l2 regularization/weight decay over the non-excluded parameters
        """
        assert self.regularized_network is not None

        # Retrieve non excluded parameters
        params = list(self.regularized_network.parameters())

        # Iterate over all parameters to decay
        l2_reg = None
        for W in params:
            if l2_reg is None:
                l2_reg = torch.sum(torch.pow(W, 2))
            else:
                l2_reg = l2_reg + torch.sum(torch.pow(W, 2))
        assert l2_reg is not None

        return l2_reg


def get_loss(config_loss, device=None, config_args=None, li_flag=False):
    """
    Construct loss object, wrapped if there are multiple losses
    """
    loss_name = config_loss["name"]
    if loss_name == "multitask":
        loss = WrapperLoss(config_args=config_args, device=device, config_loss=config_loss, li_flag=li_flag)
    elif loss_name in DICT_LOSS_STANDARD:
        loss = DICT_LOSS_STANDARD[loss_name](
            config_loss=config_loss, config_args=config_args, device=device
        )
    else:
        raise Exception(f"Loss {loss_name} not implemented")
    loss.print_details()
    return loss
