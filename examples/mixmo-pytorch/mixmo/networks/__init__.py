"""
Networks used in the main paper
"""

from mixmo.utils.logger import get_logger

from mixmo.networks import resnet, wrn, resnet_rawfoot, pretrain_rawfoot
import torch

LOGGER = get_logger(__name__, level="DEBUG")


def get_network(config_network, config_args):
    """
        Return a new instance of network
    """
    # Available networks for tiny
    if config_args["data"]["dataset"].startswith('tinyimagenet') or config_args["data"]["dataset"].startswith('imagenet'):
        print(config_network["name"])
        if config_network["name"]=='resnet':
            network_factory = resnet.resnet_network_factory
        elif config_network["name"]=='wideresnet':
            network_factory = wrn.wrn_network_factory
    elif config_args["data"]["dataset"].startswith('cifar'):
        network_factory = wrn.wrn_network_factory
    elif config_args["data"]["dataset"].startswith('rawfoot'):
        if config_network["name"]=='resnet':
            network_factory = resnet_rawfoot.resnet_network_factory
        elif config_network["name"]=='pretrain':
            network_factory = pretrain_rawfoot.pretrain_network_factory
    elif config_args["data"]["dataset"].startswith('marioiggy'):
        network_factory = resnet.resnet_network_factory ##Use resnet for tinyimagenet for simplcity
    else:
        raise NotImplementedError

    LOGGER.warning(f"Loading network: {config_network['name']}")
    return torch.nn.DataParallel(network_factory[config_network["name"]](
        config_network=config_network,
        config_args=config_args))
