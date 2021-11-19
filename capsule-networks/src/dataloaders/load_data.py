"""
PyTorch implementation of Capsule Networks

Dynamic Routing Between Capsules: https://arxiv.org/abs/1710.09829

Author: Riccardo Renzulli
University: Università degli Studi di Torino, Department of Computer Science
"""

import torch
import torchvision
import random
import math
import numpy as np
import ops.utils as utils
from numpy.core.numeric import indices
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(config):
    """Return train and test loaders for config.dataset."""
    if config.dataset == "mnist":
        return get_dataloaders_mnist(config)
    elif config.dataset == "fashion-mnist":
        return get_dataloaders_fashion_mnist(config)
    elif config.dataset == "cifar10":
        return get_dataloaders_cifar10(config)


def get_dataloaders_mnist(config):
    """Return train, val and test loaders for MNIST."""
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                size=(config.input_height, config.input_width), padding=config.shift_pixels),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.CenterCrop(size=(config.input_height, config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(size=(config.input_height, config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    train_dataset = datasets.MNIST(root="../data/raw/mnist",
                                   train=True,
                                   transform=train_transform,
                                   download=True)

    valid_dataset = datasets.MNIST(root="../data/raw/mnist",
                                   train=True,
                                   transform=test_transform,
                                   download=True)

    indices = list(range(len(train_dataset)))
    split = 5500
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))
    test_dataset = datasets.MNIST(root="../data/raw/mnist",
                                  train=False,
                                  transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             worker_init_fn=lambda id: utils.set_seed(42))
    return train_loader, valid_loader, test_loader

def get_dataloaders_fashion_mnist(config):
    """Return train, val and test loaders for Fashion-MNIST."""
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                size=(config.input_height, config.input_width), padding=config.shift_pixels),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.CenterCrop(size=(config.input_height, config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(size=(config.input_height, config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    train_dataset = datasets.FashionMNIST(root="../data/raw/fashion-mnist",
                                          train=True,
                                          transform=train_transform,
                                          download=True)

    indices = list(range(len(train_dataset)))
    split = 5500
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    valid_dataset = datasets.FashionMNIST(root="../data/raw/fashion-mnist",
                                          train=True,
                                          transform=test_transform,
                                          download=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))
    test_dataset = datasets.FashionMNIST(root="../data/raw/fashion-mnist",
                                         train=False,
                                         transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             worker_init_fn=lambda id: utils.set_seed(42))
    return train_loader, valid_loader, test_loader


def get_dataloaders_cifar10(config):
    """Return train, val and test loaders for CIFAR10."""
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                size=(config.input_height, config.input_width), padding=config.shift_pixels),
            transforms.RandomHorizontalFlip(p=0.5),
            #https://github.com/Sarasra/models/blob/master/research/capsules/input_data/cifar10/cifar10_input.py#L80
            transforms.ColorJitter(brightness=0.25, contrast=(0.2, 1.8), saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
    ])
    else:
        train_transform = transforms.Compose([
            transforms.CenterCrop(
                size=(config.input_height, config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(
            size=(config.input_height, config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    train_dataset = datasets.CIFAR10(root="../data/raw/cifar10",
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    indices = list(range(len(train_dataset)))
    split = 5500
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    valid_dataset = datasets.CIFAR10(root="../data/raw/cifar10",
                                     train=True,
                                     transform=test_transform,
                                     download=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))
    test_dataset = datasets.CIFAR10(root="../data/raw/cifar10",
                                    train=False,
                                    transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             worker_init_fn=lambda id: utils.set_seed(42))
    return train_loader, valid_loader, test_loader