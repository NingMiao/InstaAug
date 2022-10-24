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
        self.bias=nn.Parameter(torch.zeros(*shape).type(torch.float))

    def forward(self):
        return self.bias

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out += shortcut
        return out


class LogitNet(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        
        config_network={"depth":18, "widen_factor":1}
        self.config_network = config_network
        self.main_layer=4
            
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_output_network()
        self._init_weights_resnet()
        
        LOGGER.warning("Features dimension: {features_dim}".format(features_dim=self.features_dim))

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
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
            18: [2 for i in range(self.main_layer)],#!
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            8,
            8 * widen_factor, 16 * widen_factor,
            32 * widen_factor,  32 * widen_factor
        ]
        
        
    def _init_first_layer(self):
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
        
        self.layer4= self._make_layer(self._block, planes=self._nChannels[4],
                                       blocks=self._layers[3], stride=2)
                        
        self.features_dim = self._nChannels[-1] * self._block.expansion
    
    def _init_output_network(self):
        
        self.output_layer=nn.Conv2d(self.features_dim, 1, kernel_size=1, stride=1, padding=0, bias=False)
                
            
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
        #for item in self.output_conv_list:
        #    item[0].weight.data*=0.3#!Stablize training.


    def forward(self, x): 
                
        merged_representation = self._forward_first_layer(x)
        
        x4=self._forward_core_network(merged_representation)
        
        out=self.output_layer(x4)
        
        return out/10.0
    
    def _forward_first_layer(self, pixels):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x0=x
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4
    
def get_scope(input_size=224):
    from copy import deepcopy as copy

    def padding(scope, paddings=1):
        scope_new=copy(scope)
        step_size=scope[1][0]-scope[0][0]
        for i in range(paddings):
            scope_new.insert(0, [scope_new[0][0]-step_size, scope_new[0][1]-step_size])
            scope_new.append([scope_new[-1][0]+step_size, scope_new[-1][1]+step_size])
        return scope_new
    def conv(scope, convs=1):
        #After padding
        scope_new=[]
        for i in range(convs, len(scope)-convs):
            scope_new.append([scope[i-convs][0], scope[i+convs][1]])
        return scope_new
    def stride(scope, strides=2):
        scope_new=copy(scope)
        scope_new=scope_new[::strides]
        return scope_new
    scope_input=[[i, i+1] for i in range(input_size)]
    scope_v0=stride(conv(padding(scope_input)))
    scope_v1=conv(padding(conv(padding(conv(padding(conv(padding(scope_v0))))))))
    scope_v2=conv(padding(conv(padding(conv(padding(stride(conv(padding(scope_v1))))))),2),2)
    scope_v3=conv(padding(conv(padding(conv(padding(stride(conv(padding(scope_v2)))))))))
    scope_v4=conv(padding(conv(padding(conv(padding(stride(conv(padding(scope_v3)))))))))
    
    return scope_v4


class ImageLogit(nn.Module):
    def __init__(self, input_size=224, sizes=[0.4, 0.6, 0.8, 1.0], strides=[2,1,1,1], bias=[0,0,0,0], device='cpu', *args, **kwargs):
        nn.Module.__init__(self)
        self.net=LogitNet()
        self.input_size=input_size
        self.sizes=sizes
        self.strides=strides
        self.device=device
        
        self.Adjust_bias=AddBias([4])
        
        #self.adjust_bias=nn.Parameter(torch.tensor(bias,dtype=torch.float32)).to(device)
        self.number_balance_bias = (-2*torch.log(torch.tensor([11, 10, 4, 1]).to(torch.float32))+torch.tensor(bias,dtype=torch.float32)).to(device)
        #self.bias=self.adjust_bias+self.number_balance_bias
        #Don't use nn.Parameter, which leads to frequent cpu-tpu communication
        
        self._init_resize_mats()
        
        self._remove_margin()
        self._build_indexing_mat()
        
    def _init_resize_mats(self):
        self.out_size=[int(self.input_size/size) for size in self.sizes]
        resize_mats=[self._get_resize_mat(self.input_size, out_size) for out_size in self.out_size]
        self.resize_mats=[(mat.T, mat) for mat in resize_mats]
        
    def _get_resize_mat(self, in_size, out_size):
        #out_size > in_size
        mat=np.zeros([in_size, out_size], dtype=np.float32)
        r=(in_size-1)/(out_size-1)
        for i in range(out_size-1):
            j_float=r*i
            j_floor=int(j_float)
            j_ceil=int(j_float)+1
            mat[j_floor, i]=j_ceil-j_float
            mat[j_ceil, i]=j_float-j_floor
        mat[-1, -1]=1
        return torch.tensor(mat).to(self.device)
   
    def forward(self, imgs):
        adjust_bias=self.Adjust_bias()*5
        bias=adjust_bias+self.number_balance_bias
        
        resized_imgs=[torch.matmul(torch.matmul(mat_1, imgs), mat_2) for (mat_1, mat_2) in self.resize_mats]
        self.saved_resized_img=self._cat_img(resized_imgs) 

        logits = [self.net(resized_imgs[i])[:,0, self.non_black_indexes[i][0]:self.non_black_indexes[i][1]:self.strides[i], self.non_black_indexes[i][0]:self.non_black_indexes[i][1]:self.strides[i]] for i in range(len(bias))]

        logits = [logits[i]+bias[i] for i in range(len(bias))]
        logits_reshape=[logit.reshape(logit.shape[0], -1) for logit in logits]
        logits_tensor=torch.cat(logits_reshape, dim=-1)
        return logits_tensor
        
    def _cat_img(self, imgs):
        img_0=imgs[0]
        img_1=torch.cat([imgs[1], imgs[0][:,:,:self.out_size[1],:self.out_size[0]-self.out_size[1]]], dim=-1)
        img_2=torch.cat([imgs[2], imgs[0][:,:,:self.out_size[2],:self.out_size[0]-self.out_size[2]]], dim=-1)
        img_3=torch.cat([imgs[3], imgs[0][:,:,:self.out_size[3],:self.out_size[0]-self.out_size[3]]], dim=-1)
        return torch.cat([img_0, img_1, img_2, img_3], dim=-2)
    
    def get_image(self, indexes):
        n_copies=int(len(indexes)/self.saved_resized_img.shape[0])
        resize_img = torch.tile(self.saved_resized_img, [n_copies,1,1,1]) # [batch*n_copies, 3,  le, se]
        mat_l=torch.index_select(self.mat_l, 0, indexes) # [batch*n_copies, 224, le]
        mat_r=torch.index_select(self.mat_r, 0, indexes) # [batch*n_copies, se, 224]
        #Left
        mat_l = torch.unsqueeze(mat_l, 1) # [batch*n_copies, 1, 224, le]
        img_left=torch.matmul(mat_l, resize_img) #[batch*n_copies, 3, 224, se]
        mat_r = torch.unsqueeze(mat_r, 1)  # [batch*n_copies,1, se, 224]

        img_out=torch.matmul(img_left, mat_r) # [batch*n_copies,1, 224, 224]
        
        return img_out
    
    def _remove_margin(self):
        scopes=[[[sizes[0]-1, sizes[1]] for sizes in get_scope(self.out_size[i])] for i in range(len(self.out_size))] #sizes[0]-1 is to make size equal 224
        non_black_indexes=[]
        non_black_scopes=[]
        
        for i in range(len(scopes)):
            flag=0
            start_idx=None
            end_idx=None
            for j in range(len(scopes[i])):
                if scopes[i][j][0]>=0 and scopes[i][j][1]<=self.out_size[i]:
                    if flag==0:
                        start_idx=j
                        flag=1
                else:
                    if flag==1:
                        end_idx=j
                        break
            if start_idx and end_idx:
                non_black_indexes.append([start_idx, end_idx])
                non_black_scopes.append([scopes[i][j] for j in range(start_idx, end_idx)])
            else:
                non_black_indexes.append([])
                non_black_scopes.append([])
        self.non_black_indexes=non_black_indexes
        self.non_black_scopes=non_black_scopes
                
        idx2detail=[]
        for i in range(len(scopes)):
            for j in range(0, len(self.non_black_scopes[i]), self.strides[i]):
                for k in range(0, len(self.non_black_scopes[i]), self.strides[i]):
                    idx2detail.append([i, self.non_black_scopes[i][j][0], self.non_black_scopes[i][k][0]])
        self.idx2detail=idx2detail
    
    def _build_indexing_mat(self):
        #Can further save memory if memory is limited
        l=len(self.idx2detail)
        le=sum(self.out_size)
        se=self.out_size[0]
        #Right mat
        mat_r=torch.zeros([l, se, 224], dtype=torch.float32)
        for i in range(l):
            for j in range(224):
                mat_r[i, self.idx2detail[i][2]+j, j]=1
        #Left mat
        mat_l=torch.zeros([l, 224, le], dtype=torch.float32)
        for i in range(l):
            start=self.idx2detail[i][1]
            for k in range(self.idx2detail[i][0]):
                start+=self.out_size[k]
            for j in range(224):
                mat_l[i, j, start+j]=1
        self.mat_l=mat_l.to(self.device)
        self.mat_r=mat_r.to(self.device)
    
if __name__=='__main__':
    net=ImageLogit()
    x=torch.randn([100,3,224,224])*0
    #x[0,0,0,0]=10
    x[0,:,:112,:]+=10
    
    for i in range(10):
        feature=net(x)
        net.get_image([88]*100)[0]
    print(feature.shape)
    #print(feature[0])
    print(feature[0,:121].reshape(11,11))
    #print(feature[0,121:221].reshape(10,10))
    #print(feature[0,221:237].reshape(4,4))
    #print(feature[0,237:].reshape(1,1))
    #print(net.get_image([88]*10)[0])
    #print([item.shape for item in net.parameters()])
    #print(net.get_image(torch.arange(10)))
    #print(net.idx2detail)
    #print(get_scope(input_size=215))