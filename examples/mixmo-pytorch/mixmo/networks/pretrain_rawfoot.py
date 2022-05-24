#! Maybe support MIMO settings now!

import torch
import torch.nn as nn
from torch.nn import functional as F

from mixmo.augmentations import mixing_blocks
from mixmo.utils import torchutils
from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")

BATCHNORM_MOMENTUM_PREACT = 0.1

def feature_module(device='cpu'):
    import torchvision
    model=torchvision.models.resnet18(pretrained=True, progress=True)
    
    def feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x
    model.feature=feature.__get__(model)
    model.eval()
    model.to(device)
    return model.feature

class PretrainNet(nn.Module):
    def __init__(self, config_network,config_args):
        super(PretrainNet, self).__init__()
        class_num=config_args["data"]["num_classes"]
        self.fc_layer=torch.nn.Linear(512, class_num)
        if next(self.fc_layer.parameters()).is_cuda:
            device='cuda'
        else:
            device='cpu'
        device='cuda'
        self.feature=feature_module(device=device)
        
    def forward(self, x, feature=False):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            x = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            x = x
        
        extracted_feature=self.feature(x).squeeze(-1).squeeze(-1)
        output=self.fc_layer(extracted_feature)
        dict_output= {"logits": output, "logits_0": output}
        
        if feature:
            return dict_output, extracted_feature
        return dict_output


pretrain_network_factory = {
    # For Rawfoot
    "pretrain": PretrainNet,
}
