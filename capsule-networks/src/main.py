"""
PyTorch implementation of Capsule Networks

Dynamic Routing Between Capsules: https://arxiv.org/abs/1710.09829

Author: Riccardo Renzulli
University: Università degli Studi di Torino, Department of Computer Science
"""

import os
import torch
import logging
import json
import argparse
import loss.capsule_loss as cl
import ops.utils as utils
import torch.nn as nn
from os.path import dirname, abspath
from ops.utils import save_args
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from train import train
from test import test
from torch.utils.tensorboard import SummaryWriter
from layers.capsule import CapsClass2d
import warnings
warnings.filterwarnings("ignore")

def train_test_caps(config):
    experiment_folder = utils.create_experiment_folder(config, config.seed)

    utils.set_seed(config.seed)
    test_base_dir = "../results/" + config.dataset + "/" + config.model + "/" + experiment_folder

    logdir = test_base_dir + "/logs/"
    checkpointsdir = test_base_dir + "/checkpoints/"
    runsdir = test_base_dir + "/runs/"
    imgdir = test_base_dir + "/images/"

    # Make model checkpoint directory
    if not os.path.exists(checkpointsdir):
        os.makedirs(checkpointsdir)

    # Make log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Make img directory
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    # Set logger path
    utils.set_logger(os.path.join(logdir, "model.log"))

    # Get dataset loaders
    train_loader, valid_loader, test_loader = get_dataloader(config)

    # Enable GPU usage
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device(config.cuda_device)
    else:
        device = torch.device("cpu")

    caps_model = VectorCapsNet(config, device)

    utils.summary(caps_model, config)

    caps_criterion = cl.CapsLoss(config.caps_loss,
                                 config.margin_loss_lambda,
                                 config.reconstruction_loss_lambda,
                                 config.batch_averaged,
                                 config.reconstruction is not None,
                                 config.m_plus,
                                 config.m_minus,
                                 config.m_min,
                                 config.m_max,
                                 device)

    if config.optimizer == "adam":
        caps_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr)
    else:
        caps_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr)
    caps_scheduler = torch.optim.lr_scheduler.ExponentialLR(caps_optimizer, config.decay_rate)

    caps_model.to(device)

    for state in caps_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Print the model architecture and parameters
    utils.summary(caps_model, config)

    # Save current settings (hyperparameters etc.)
    save_args(config, test_base_dir)

    # Writer for TensorBoard
    writer = None
    if config.tensorboard:
        writer = SummaryWriter(runsdir)

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    logging.info("Device: {}".format(device))

    logging.info("Initial learning rate: {:.4f}".format(caps_scheduler.get_last_lr()[0]))
    logging.info("Number of routing iterations: {}".format(config.num_routing_iterations))

    best_loss = float('inf')

    epoch = 0
    best_epoch = 0
    training = True
    while training:
        # Start training
        logging.info("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))
        train(logging, config, train_loader, caps_model, caps_criterion, caps_optimizer, caps_scheduler, writer, epoch, device)

        # Start validation
        val_loss, val_acc = test(logging, config, valid_loader, caps_model, caps_criterion, writer, epoch, device,
                         imgdir, split="validation")
        # Start testing
        test_loss, test_acc = test(logging, config, test_loader, caps_model, caps_criterion, writer, epoch, device, imgdir, split="test")

        
        if writer:
            writer.add_scalar('routing/iterations', caps_model.classCaps.num_iterations, epoch)
            writer.add_scalar('lr', caps_scheduler.get_last_lr()[0], epoch)

        formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))
        checkpoint_filename = "epoch_{}".format(formatted_epoch)

        if val_loss < best_loss:
            utils.save_checkpoint({
                "epoch": epoch,
                "routing_iterations": caps_model.classCaps.num_iterations,
                "state_dict": caps_model.state_dict(),
                "metric": config.monitor,
                "optimizer": caps_optimizer.state_dict(),
                "scheduler": caps_scheduler.state_dict(),
            }, True, checkpointsdir, checkpoint_filename)
            best_epoch = epoch
            best_loss = val_loss

        # Save current epoch checkpoint
        utils.save_checkpoint({
            "epoch": epoch,
            "routing_iterations": caps_model.classCaps.num_iterations,
            "state_dict": caps_model.state_dict(),
            "metric": config.monitor,
            "optimizer": caps_optimizer.state_dict(),
            "scheduler": caps_scheduler.state_dict(),
        }, False, checkpointsdir, checkpoint_filename, config.dataset=="mnist" and config.reconstruction=="None" and config.seed==42)
        epoch += 1
        if epoch - best_epoch > config.patience:
            training = False
    if writer:
        writer.close()

def main(config):
    for k in range(len(config.seeds)):
        config.seed = config.seeds[k]
        train_test_caps(config)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    config = utils.DotDict(json.load(open(args.config)))
    main(config)