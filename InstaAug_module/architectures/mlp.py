import torch
import torch.nn as nn
from copy import deepcopy as copy

class Mlp(nn.Module):
    """
    Vanilla Mlp
    """

    def __init__(self, num_inputs, num_targets, hidden_list=[100,20]):
        super().__init__()
        hidden_list=copy(hidden_list)
        hidden_list.append(num_targets)
        hidden_list.insert(0, num_inputs)
        layers=[]
        for i in range(len(hidden_list)-1):
            layers.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
            if i!=len(hidden_list)-2:
                layers.append(nn.LeakyReLU(0.2))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

if __name__=='__main__':
    mlp=Mlp(10, 5)
    r=torch.randn([20,10])
    print(mlp(r).shape)