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
import numpy as np

from .mixmo_utils import torchutils
#from .mixmo_utils.logger import get_logger

#LOGGER = get_logger(__name__, level="DEBUG")
class LOGGER:
    @staticmethod
    def warning(string):
        print(string)

BATCHNORM_MOMENTUM_PREACT = 0.1

class AddBias(nn.Module):
    def __init__(self, shape):
        super(AddBias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(1, *shape).type(torch.float))

    def forward(self, x):
        return x+self.bias

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


class PreActResFeatureNet_Imagenet(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, output_layer=[3, -1], output_dims=[], input_size=224, main_layer=5, *args, **kwargs):
        nn.Module.__init__(self)
        
        config_network={"depth":18, "widen_factor":1}
        config_args={"data":{"num_classes":200}, "num_members":1}
        if 'num_targets' in kwargs:
            config_args['data']["num_classes"]=kwargs['num_targets']
        self.config_network = config_network
        self.config_args = config_args
        self.input_size=input_size
        self.main_layer=main_layer
        
        if output_dims==[]:
            self.output_dims=[1 for layer in output_layer]
        else:
            self.output_dims=output_dims
            
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self.output_layer=output_layer
        self._init_output_network()
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
            18: [1 for i in range(self.main_layer)],
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
            8,
            8 * widen_factor, 32 * widen_factor,
            32 * widen_factor
        ]
        for i in range(3, self.main_layer):
            self._nChannels.append(32 * widen_factor)
        
        
    def _init_first_layer(self):
        assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)#! The channel is changed to 6

    def _init_core_network(self):
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
        
        if self.main_layer>=4:
            self.layer4= self._make_layer(self._block, planes=self._nChannels[4],
                                       blocks=self._layers[3], stride=2)
        if self.main_layer>=5:
            self.layer5= self._make_layer(self._block, planes=self._nChannels[5],
                                       blocks=self._layers[4], stride=2)
        
        
        self.features_dim = self._nChannels[-1] * self._block.expansion
    
    def _init_output_network(self):
        
        #New for crop_and_color
        self.output_conv_list=[]
        for i in range(len(self.output_layer)):
            layer=self.output_layer[i]
            
            size_layers=[int(np.ceil(self.input_size/2))]
            size_layers.append(size_layers[-1])
            for j in range(1, self.main_layer):
                size_layers.append(int(np.ceil(size_layers[-1]/2)))
            if layer > 0:
                feature_pre=nn.Conv2d(
                self._nChannels[layer], self.output_dims[i], kernel_size=1, stride=1, padding=0, bias=False)
                size=size_layers[layer]
                feature=nn.Sequential(feature_pre, AddBias([self.output_dims[i], size, size]))
                
                
            if layer==-1:
                feature_pre=nn.Conv2d(
                self._nChannels[layer], self.output_dims[i], kernel_size=size_layers[-1], stride=1, padding=0, bias=False)
                feature=nn.Sequential(feature_pre, AddBias([self.output_dims[i], 1, 1]))
            
            
            #exec('self.conv_feature'+'66726'+' =feature')#!
            exec('self.conv_feature'+str(i)+' =feature')
            
            self.output_conv_list.append(feature)
            self.output_conv_list=torch.nn.ModuleList(self.output_conv_list)#!
        
        ##Old for crop only
        #layer=0
        #self.conv_feature0=nn.Conv2d(
        #        self._nChannels[layer], 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        #layer=1
        #self.conv_feature1=nn.Conv2d(
        #        self._nChannels[layer], 1, kernel_size=1, stride=1, padding=0, bias=True)
        #layer=2
        #self.conv_feature2=nn.Conv2d(
        #        self._nChannels[layer], 1, kernel_size=1, stride=1, padding=0, bias=True)
        #layer=3
        #self.conv_feature3=nn.Conv2d(
        #        self._nChannels[layer], 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        #self.conv_feature_global=nn.Conv2d(
        #        self._nChannels[layer], 1, kernel_size=8, stride=1, padding=0, bias=True)
        
        #conv_list=[self.conv_feature0, self.conv_feature1, self.conv_feature2, self.conv_feature3, self.conv_feature_global]
        
        
        #self.output_conv_list=[]
        #for i in self.output_layer:
        #    self.output_conv_list.append(conv_list[i])
    
    
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

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            torchutils.weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)
        for item in self.output_conv_list:
            item[0].weight.data*=0.3#!Stablize training.


    def forward(self, x): 

        merged_representation = self._forward_first_layer(x)
        
        map_list=self._forward_core_network(merged_representation)
        
        out_list=[]
        for i in range(len(self.output_layer)):
            out=self.output_conv_list[i](map_list[self.output_layer[i]])
            #!out_list.append(out[:,0])#??
            out_list.append(out)#??
        
        return out_list      
    
    def _forward_first_layer(self, pixels):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x0=x
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        out_list=[x0, x1, x2, x3]
        if self.main_layer>=4:
            x4=self.layer4(x3)
            out_list.append(x4)
        if self.main_layer>=4:
            x5=self.layer5(x4)
            out_list.append(x5)  
        return out_list
    
    def center(self, interval=False):
                
        n_pre=self.input_size
        c_pre=torch.zeros([n_pre,n_pre,2])
        base=2/n_pre
        for i in range(n_pre):
            for j in range(n_pre):
                c_pre[i,j,0]=-1+base/2+i*base
                c_pre[i,j,1]=-1+base/2+j*base
        
        c0=(c_pre[0::2, 0::2]+c_pre[1::2, 1::2])/2 #New version, get the center rather than left up corner
        c1=c0
        c2=(c1[0::2, 0::2]+c1[1::2, 1::2])/2
        c3=(c2[0::2, 0::2]+c2[1::2, 1::2])/2
        c_global=torch.zeros([1,1,2])
        center_dict={0: c0, 1: c1, 2: c2, 3: c3, -1: c_global}
        if self.main_layer>=4:
            c4=(c3[0::2, 0::2]+c3[1::2, 1::2])/2
            center_dict[4]=c4
        if self.main_layer>=5:
            c5=(c4[0::2, 0::2]+c4[1::2, 1::2])/2
            center_dict[5]=c5
        
        
        
        centers=[center_dict[l] for l in self.output_layer]
        if interval:
            center_interval_dict={i: center_dict[i][1,0,0]-center_dict[i][0,0,0] for i in range(self.main_layer+1)}
            center_interval_dict[-1]=0.1
            center_intervals=[center_interval_dict[l] for l in self.output_layer]
            return centers, center_intervals
        else:        
            #return centers    
            center_interval_dict={i: 0.0 for i in range(self.main_layer+1)}
            center_interval_dict[-1]=0.0
            center_intervals=[center_interval_dict[l] for l in self.output_layer]
            return centers, center_intervals
    
    def scope(self, ranges=False):
        
        n_pre=self.input_size
        s_pre=2/n_pre
        
        s0=s_pre+s_pre
        s1=s0+s_pre*2+s_pre*2
        s2=s1+s_pre*2+s_pre*4
        s3=s2+s_pre*4+s_pre*8
        scope_dict={0: s0, 1: s1, 2: s2, 3: s3, -1: 1}
        if self.main_layer>=4:
            s4=s3+s_pre*8+s_pre*16
            scope_dict[4]=s4
        if self.main_layer>=5:
            s5=s4+s_pre*16+s_pre*32
            scope_dict[5]=s5
        scopes=[scope_dict[l] for l in self.output_layer]
        
        if ranges:
            ##Has at least two output layers, which includes -1
            scope_ranges=[]
            scope_ranges.append([max(0.3, scopes[0]-0.1), (scopes[0]+scopes[1])/2])
            for i in range(1, len(scopes)-1):
                min_range=(scopes[i]+scopes[i-1])/2
                max_range=(scopes[i]+scopes[i+1])/2
                scope_ranges.append([min_range, max_range])
            scope_ranges.append([(scope_ranges[-1][1]+1)/2, 1])
            return scopes, scope_ranges
        else:
            #return scopes
            scope_ranges=[]
            for i in range(len(scopes)):
                min_range=(scopes[i]+scopes[i])/2
                max_range=(scopes[i]+scopes[i])/2
                scope_ranges.append([min_range, max_range])
            return scopes, scope_ranges
    
    def mask_for_no_padding(self, max_black_ratio=0.2):
        centers, _=self.center()
        scopes, _=self.scope()
        paddings=[]
        for i in range(len(centers)):
            center=centers[i]
            scope=scopes[i]
            for j in range(center.shape[0]):
                area_non_black=(center[j,j,0]+1+scope)**2
                area_whole=(scope*2)**2
                blank_ratio=1-area_non_black/area_whole
                if blank_ratio<=max_black_ratio:
                    paddings.append(j)
                    break
        return paddings

if __name__=='__main__':
    net=PreActResFeatureNet(input_size=224, main_layer=5, output_layer=[1,2,3,4,5,-1], output_dims=[1,1,1,1,1,1])
    x=torch.randn([10,3,224,224])
    #scopes: [0.05357142857142857, 0.10714285714285714, 0.21428571428571427, 0.42857142857142855, 0.8571428571428571, 1]
    #net=PreActResFeatureNet(input_size=64, main_layer=3, output_layer=[1,2,3,-1], output_dims=[1,1,1,1])
    #x=torch.randn([10,3,64,64])
    
    feature=net(x)
    print(net.scope())
    print([y.shape for y in feature])