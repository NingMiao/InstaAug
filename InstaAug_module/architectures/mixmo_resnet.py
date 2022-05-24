'''
Resnet for tinyImagenet dataset.
Adapted from
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mixmo_utils import torchutils
#from .mixmo_utils.logger import get_logger

#LOGGER = get_logger(__name__, level="DEBUG")
class LOGGER:
    @staticmethod
    def warning(string):
        print(string)

BATCHNORM_MOMENTUM_PREACT = 0.1

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

#Temporarily use fixed parameters here


class PreActResNet(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        
        config_network={"depth":18, "widen_factor":1}
        config_args={"data":{"num_classes":200}, "num_members":1}
        if 'num_targets' in kwargs:
            config_args['data']["num_classes"]=kwargs['num_targets']
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()
        LOGGER.warning("Features dimension: {features_dim}".format(features_dim=self.features_dim))

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = self.config_args["data"]["num_classes"]
        self.depth = self.config_network["depth"]
        self._init_block(widen_factor=self.config_network["widen_factor"])

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        #!self._nChannels = [
        #    64,
        #    64 * widen_factor, 128 * widen_factor,
        #    256 * widen_factor, 512 * widen_factor
        #]
        self._nChannels = [
            32,
            32 * widen_factor, 64 * widen_factor,
            128 * widen_factor, 256 * widen_factor
        ]
        
        
    def _init_first_layer(self):
        assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            torchutils.weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x, output_feature=False):

        merged_representation = self._forward_first_layer(x)
        
        if output_feature:
            extracted_features=self._forward_core_network(merged_representation, layer=-1)
            #print(extracted_features.shape)
            return extracted_features.detach() #!
        
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output        
    
    def _forward_first_layer(self, pixels):
        return self.conv1(pixels)

    def _forward_core_network(self, x, layer=-1):
        if layer==0:
            return x.view(x.size(0), -1)
        x = self.layer1(x)
        if layer==1:
            return x.view(x.size(0), -1)
        x = self.layer2(x)
        if layer==2:
            return x.view(x.size(0), -1)
        x = self.layer3(x)
        if layer==3:
            return x.view(x.size(0), -1)
        x = self.layer4(x)
        if layer==4:
            return x.view(x.size(0), -1)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        return x



resnet_network_factory = {
    # For TinyImageNet
    "resnet": PreActResNet,
}

if __name__=='__main__':
    net=PreActResNet()