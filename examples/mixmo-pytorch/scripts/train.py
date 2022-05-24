import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
import os
from shutil import copyfile, rmtree
import click
import torch

from mixmo.loaders import get_loader
from mixmo.learners.learner import Learner
from mixmo.utils import (misc, config, logger)
from scripts.evaluate import evaluate

import sys
sys.path.insert(0, '../../')
from InstaAug_module import learnable_invariance

LOGGER = logger.get_logger(__name__, level="DEBUG")



def parse_args():
    parser = argparse.ArgumentParser()

    # shared params
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml", required=True)
    parser.add_argument("--dataplace", "-dp", type=str, default=None, help="Parent folder to data", required=True)
    parser.add_argument("--saveplace", "-sp", type=str, default=None, help="Parent folder to save", required=True)
    parser.add_argument("--gpu", "-g", default="", type=str, help="Selecting gpu. If not exists, then cpu")
    parser.add_argument("--debug", type=int, default=0, help="Debug mode: 0, 1 or 2. The more the more debug.")

    # specific params
    parser.add_argument(
        "--from_scratch",
        "-f",
        action="store_true",
        default=False,
        help="Force training from scratch",
    )
    parser.add_argument(
        "--seed",
        default=config.cfg.RANDOM.SEED,
        type=int,
        help="Random seed",
    )
    
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--resume_classifier_only",action="store_true",default=False,)
    
    #weight
    parser.add_argument("--h_weight", type=float, default=0.01)
    parser.add_argument("--s_weight", type=float, default=0.01)
    parser.add_argument("--v_weight", type=float, default=0.01)
    
    #weight
    parser.add_argument("--random_h_bound", type=float, default=0.00)
    parser.add_argument("--random_s_bound", type=float, default=0.00)
    parser.add_argument("--random_v_bound", type=float, default=0.00)
    
    parser.add_argument("--train_name", type=str, default='14')
    
    #Learnable augmentation
    parser.add_argument("--Li_config_path", type=str, default='')
    parser.add_argument("--max_tolerance", type=int, default=1000)
    
    # parse
    args = parser.parse_args()
    misc.print_args(args)
    return args

def transform_args(args):
    # shared transform
    config_args = misc.load_config_yaml(args.config_path)
    config_args["training"]["output_folder"] = output_folder = misc.get_output_folder_from_config(
        saveplace=args.saveplace, config_path=args.config_path, info=args.info
    )
    config.cfg.DEBUG = args.debug
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # recover epoch and checkpoints
    if os.path.exists(output_folder):
        if args.from_scratch or (output_folder not in args.checkpoint):
            rmtree(output_folder)
            os.mkdir(output_folder)
    
    if args.from_scratch:
        checkpoint=None
    else:
        if args.checkpoint=='':
            checkpoint = misc.get_previous_ckpt(output_folder)
        else:
            checkpoint=args.checkpoint
    
    if not os.path.exists(output_folder.split('/')[0]):
        os.mkdir(output_folder.split('/')[0])
        
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # seed
    config.cfg.RANDOM.SEED = args.seed
    misc.set_determ(config.cfg.RANDOM.SEED)
    
    config_args['resume_classifier_only']=args.resume_classifier_only
    config_args['weights']=[args.h_weight, args.s_weight, args.v_weight]
    config_args['random_bounds']=[args.random_h_bound, args.random_s_bound, args.random_v_bound]
    config_args['max_tolerance']=args.max_tolerance
    return config_args, device, checkpoint


def train(config_args, Li_configs, device, checkpoint, dataplace, train_name=None):

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args, dataplace=dataplace, train_name=train_name)
    
    #Load Li
    if Li_configs['li_flag']:
        Li=learnable_invariance(Li_configs)
    else:
        Li=None
        
    # Set learner
    learner = Learner(
        config_args=config_args,
        Li_configs=Li_configs, #?
        dloader=dloader,
        device=device,
        Li=Li, #!
    )
    
    #!learner.model_wrapper.weights=config_args['weights']
    
    # Resume existing model or from pretrained one
    if config_args['resume_classifier_only']:
        learner.load_checkpoint(
            checkpoint, include_optimizer=False, return_epoch=False)
        start_epoch=1
    elif checkpoint is not None:
        LOGGER.warning(f"Load checkpoint: {checkpoint}")
        try:
            start_epoch = learner.load_checkpoint(
                checkpoint, include_optimizer=False, return_epoch=True) + 1
        except:
            start_epoch=0
        learner.model_wrapper.optimizer.param_groups[0]['initial_lr']=0.002
        try:
            learner.model_wrapper.optimizer.param_groups[1]['initial_lr']=1e-5
        except:
            pass
        config.cfg.RANDOM.SEED = config.cfg.RANDOM.SEED - 1 + start_epoch
        misc.set_determ(config.cfg.RANDOM.SEED)
    else:
        LOGGER.info("Starting from scratch")
        start_epoch = 1

    LOGGER.info(f"Saving logs in: {config_args['training']['output_folder']}")

    # Start training
    _config_name = os.path.split(
        os.path.splitext(config_args['training']['config_path'])[0])[-1]

    try:
        epoch = start_epoch
        for epoch in range(start_epoch, config_args["training"]["nb_epochs"] + 1):
            LOGGER.debug(f"Epoch: {epoch} for: {_config_name}")
            learner.dloader.traindatasetwrapper.set_ratio_epoch(
                ratioepoch=epoch / config_args["training"]["nb_epochs"]
            )
            learner.train(epoch)

    except KeyboardInterrupt:
        LOGGER.warning(f"KeyboardInterrupt for: {_config_name}")
        if not click.confirm("continue ?", abort=False):
            raise KeyboardInterrupt

    except Exception as exc:
        LOGGER.error(f"Exception for: {_config_name}")
        raise exc

    return epoch


def main_train():
    # train
    args = parse_args()
    config_args, device, checkpoint = transform_args(args)
    
    if args.Li_config_path:
        import yaml
        Li_configs=yaml.safe_load(open(args.Li_config_path,'r'))
    else:
        Li_configs={'li_flag': False}

    train(config_args, Li_configs, device, checkpoint, dataplace=args.dataplace, train_name=args.train_name)
    print('train')
    
    # test at best epoch
    best_checkpoint = misc.get_checkpoint(
        output_folder=config_args['training']['output_folder'],
        epoch="best",
    )
    #evaluate(
    #    config_args=config_args,
    #    device=device,
    #    checkpoint=best_checkpoint,
    #    tempscale=False,
    #    corruptions=False,
    #    dataplace=args.dataplace
    #)
    #LOGGER.error(f"Finish: {config_args['training']['config_path']}")


if __name__ == "__main__":
    main_train()
