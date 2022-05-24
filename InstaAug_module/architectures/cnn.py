import torch
import torch.nn as nn


def ConvBNrelu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride), nn.BatchNorm2d(out_channels), nn.ReLU())


class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ModuleAttributeWrapper(nn.Module):
    def __init__(self, module, name, value):
        super().__init__()
        self.module = module
        setattr(self, name, value)

    def forward(self, x):
        return self.module(x)


class SmallConv(nn.Module):
    """
    Very small CNN
    """

    def __init__(self, in_channels, num_targets, k=128, dropout=True):
        super().__init__()
        self.num_targets = num_targets
        featurizer = nn.Sequential(
            ConvBNrelu(in_channels, k),
            ConvBNrelu(k, k),
            ConvBNrelu(k, 2 * k),
            nn.MaxPool2d(2),  # MaxBlurPool(2*k),
            nn.Dropout2d(0.3) if dropout else nn.Sequential(),
            ConvBNrelu(2 * k, 2 * k),
            nn.MaxPool2d(2),  # MaxBlurPool(2*k),
            nn.Dropout2d(0.3) if dropout else nn.Sequential(),
            ConvBNrelu(2 * k, 2 * k),
            nn.Dropout2d(0.3) if dropout else nn.Sequential(),
            Expression(lambda u: u.mean(-1).mean(-1)),
        )
        self.featurizer = ModuleAttributeWrapper(featurizer, "output_dim", 2 * k)
        self.clf = nn.Sequential(
            nn.Linear(2 * k, num_targets),
        )
        self.net = nn.Sequential(
            self.featurizer,
            self.clf,
        )

    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(x.size(0), -1)

class SimpleConv(nn.Module):
    """ Returns a 5-layer CNN with width parameter k. """
    def __init__(self, in_channels, num_targets, k=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 0
            nn.Conv2d(in_channels, k, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.BatchNorm2d(k),
            nn.ReLU(),

            # Layer 1
            nn.Conv2d(k, k*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(k*2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(k*2, k*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(k*4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(k*4, k*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(k*8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(k*8, num_targets, bias=True)
        )

    def forward(self, x):
        return self.net(x)

    
class SimpleConv_tiny_imagenet(nn.Module):
    """ Returns a 5-layer CNN with width parameter k. """
    def __init__(self, in_channels, num_targets, k=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 0
            nn.Conv2d(in_channels, k, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.BatchNorm2d(k),
            nn.ReLU(),

            # Layer 1
            nn.Conv2d(k, k*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(k*2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(k*2, k*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(k*4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(k*4, k*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(k*8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(k*8*4, num_targets, bias=True)
        )

    def forward(self, x):
        return self.net(x)