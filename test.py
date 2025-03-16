import argparse
import os
import yaml

import numpy as np

import torch
import torch.utils.data

from torch.nn import DataParallel

from trainer import Trainer


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='ddim.yaml', type=str, help="Path to the config file")
    parser.add_argument('--seed', default=230, type=int, metavar='N', help='Seed for initializing training (default: 230)')
    parser.add_argument("--resume", default=False, type=bool, help="Resume or Not")
    
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    use_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print(f'Using {num_gpus} GPU(s)!' if use_cuda else 'Using CPU')
    
    config.device = device
    config.num_gpus = num_gpus

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(args, config)

    if num_gpus > 1:
        trainer.diffusion = DataParallel(trainer.diffusion)
        trainer.diffusion.to(device)

    trainer.test()


if __name__ == "__main__":
    main()
