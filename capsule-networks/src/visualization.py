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
import torch.nn.functional as F
import matplotlib
import pandas as pd
import numpy as np
import ops.utils as utils
import matplotlib.pyplot as plt
import loss.capsule_loss as cl
from matplotlib import gridspec
from os.path import dirname, abspath
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from test import test


def main(args):
    base_dir = dirname(dirname(abspath(__file__)))
    config = utils.DotDict(json.load(open(args.config)))
    config.batch_size = 20

    # Enable GPU usage
    if config.use_cuda & torch.cuda.is_available():
        device = torch.device(config.cuda_device)
    else:
        device = torch.device("cpu")

    _, _, test_loader = get_dataloader(config, base_dir)

    if os.path.isfile(args.checkpoint):
        logging.info("Loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        config.num_routing_iterations = checkpoint["routing_iterations"]
        config.epoch = str(checkpoint["epoch"])
        device = "cpu"
        model = VectorCapsNet(config, device)
        model.load_state_dict(checkpoint["state_dict"])
        logging.info("Loaded checkpoint '{}'".format(args.checkpoint))
    else:
        raise Exception("No checkpoint found at '{}'".format(args.checkpoint))

    model.to(device)
    model.eval()

    # Print the model architecture and parameters
    utils.summary(model, config)
    logging.info("Number of routing iterations:'{}'".format(config.num_routing_iterations))

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    logging.info("Device: {}".format(device))

    # First batch of images
    dataiter = iter(test_loader)
    dataiter.next()
    images, labels = dataiter.next()

    # Transform to one-hot indices: [batch_size, config.num_classes]
    target = F.one_hot(labels, config.num_classes)

    # Use GPU if available
    images, target = images.to(device), labels.to(device)

    # Get class caps and reconstructions
    class_caps_poses, class_caps_activations, _, reconstructions = model(images)
    reconstructions = reconstructions.view(-1, images.size(1), images.size(2), images.size(3))

    # Get predictions
    norms = torch.sqrt(torch.sum(class_caps_poses ** 2, dim=2))
    # pred: [batch_size,]
    pred = norms.max(1, keepdim=True)[1].type(torch.LongTensor).view(config.batch_size)

    # Print labels and predictions
    lab = torch.cat((labels, pred), 0)
    lab = lab.view(2, config.batch_size)

    # Show original images vs reconstructions
    fig, axs = plt.subplots(2, config.batch_size, figsize=(10, 4))
    fig.suptitle("Original vs reconstructions epoch " + config.epoch, fontsize="xx-large")
    axs[0, 0].set_ylabel("Original", fontsize="xx-large")
    axs[1, 0].set_ylabel("Reconstruction", fontsize="xx-large")

    np_images = images.squeeze().detach().numpy()
    np_reconstructions = reconstructions.squeeze().detach().numpy()

    data = [np_images, np_reconstructions]

    for i in range(2):
        for j in range(config.batch_size):
            axs[i, j].imshow(data[i][j], cmap=config.cmap)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            axs[i, j].set_xlabel(str(lab[i, j].item()), fontsize="xx-large")
    plt.tight_layout(pad=0.4)
    plt.show()

    # Show what the individual dimensions of a capsule represent
    class_caps_shape = class_caps_poses.size(-2), class_caps_poses.size(-1)
    index = 7 #8, 14, -1
    example_caps = class_caps_poses[index].unsqueeze(0) # example_caps: [1, 10, 16]
    print(torch.norm(example_caps, dim=2))
    label_caps = target[index] # label_caps: [1, 10]
    activations = class_caps_activations[index, :].view(-1, config.num_classes)
    perturb_reconstructions = []
    perturb_range = np.linspace(start=-0.05, stop=0.05, num=11)

    nrow = class_caps_shape[0] * class_caps_shape[1]
    ncol = perturb_range.size
    for dim in range(nrow):
        for perturb in perturb_range:
            example_caps_perturbed = example_caps.clone()
            example_caps_perturbed[:, :, dim] += perturb
            perturb_rec = model.decoder(example_caps_perturbed, activations)
            perturb_rec = perturb_rec.view(images.size(1), images.size(2), images.size(3))
            perturb_rec = perturb_rec.squeeze()
            perturb_reconstructions.append(perturb_rec.squeeze())


    fig = plt.figure()

    num_rows = nrow

    gs = gridspec.GridSpec(num_rows, ncol, wspace=0, hspace=0)
    perturb_reconstructions = torch.stack(perturb_reconstructions, dim=0)
    perturb_reconstructions = perturb_reconstructions.view(nrow, ncol, perturb_reconstructions.size(1),  perturb_reconstructions.size(2))
    for i in range(num_rows):
        for j in range(ncol):
            ax = plt.subplot(gs[i, j])
            np_perturb_rec = perturb_reconstructions[i,j].detach().numpy()

            ax.imshow(np_perturb_rec, cmap=config.cmap)
            ax.set_yticks([])
            ax.set_xticks([])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


#python visualize.py --config ../results/mnist/VectorCapsNet/pruning/entropy/fixed/nofreeze/loss/margin/adam/0.001/decay/pwe_201/ancient-deluge-205/caps_lower_dims_caps_types_32/3/dim_primary_caps_8_dim_class_caps_16/params.json --checkpoint ../results/mnist/VectorCapsNet/pruning/entropy/fixed/nofreeze/loss/margin/adam/0.001/decay/pwe_201/ancient-deluge-205/caps_lower_dims_caps_types_32/3/dim_primary_caps_8_dim_class_caps_16/trial1/checkpoints/epoch_867-it3-val_loss_0.027749-val_acc_0.995273_val_t-score_0.0006786448436535218-test_loss_0.027811-test_acc_0.995800-test_t-score_0.0006868507887702435-lr1.6268552406985682e-07_.pth.tar --no-pruning
def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualizations')
    parser.add_argument('--config', default=None, type=str, help='Config path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)