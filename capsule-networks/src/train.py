"""
PyTorch implementation of Capsule Networks

Dynamic Routing Between Capsules: https://arxiv.org/abs/1710.09829

Author: Riccardo Renzulli
University: Università degli Studi di Torino, Department of Computer Science
"""

import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import *
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from layers.capsule import CapsClass2d, CapsPrimary2d
import torch.nn as nn

def train(logging, config, train_loader, model, criterion, optimizer, scheduler, writer, epoch, device):
    logging.info(
        "-------------------------------------- Training epoch {} --------------------------------------".format(epoch))
    train_vector_capsnet(logging, config, train_loader, model, criterion, optimizer, scheduler, writer, epoch,
                             device)


def train_vector_capsnet(logging, config, train_loader, model, criterion, optimizer, scheduler, writer, epoch, device):
    num_batches = len(train_loader)
    tot_samples = len(train_loader.sampler)
    loss = 0
    if config.reconstruction is not None:
        margin_loss = 0
        recons_loss = 0
    precision = np.zeros(config.num_classes)
    recall = np.zeros(config.num_classes)
    f1score = np.zeros(config.num_classes)
    correct = 0

    step = epoch * num_batches + num_batches

    model.train()

    start_time = time.time()


    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            batch_size = data.size(0)
            # Store the indices for calculating accuracy
            label = target.unsqueeze(0).type(torch.LongTensor)
            global_step = batch_idx + epoch * num_batches
            max_global_step = ((config.epochs + 1) * num_batches)
            relative_step = (1. * global_step) / max_global_step

            # Transform to one-hot indices: [batch_size, 10]
            target = F.one_hot(target, config.num_classes)
            # Use GPU if available
            data, target = data.to(device), target.to(device)

            if config.reconstruction != "None":
                class_caps_poses, class_caps_activations, coupling_coefficients, reconstructions = model(data, target)

                c_loss, m_loss, r_loss = criterion(class_caps_activations, target, data, reconstructions,
                                                step=relative_step)

                loss += c_loss.item()
                margin_loss += m_loss.item()
                recons_loss += r_loss.item()
            else:
                class_caps_poses, class_caps_activations, coupling_coefficients = model(data)
                c_loss = criterion(class_caps_activations, target, step=relative_step)

                loss += c_loss.item()

            c_loss.backward()

            #Train step
            if (batch_idx + 1) % config.iter_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Count correct numbers
            # norms: [batch_size, 10, 16]
            # pred: [batch_size,]
            pred = class_caps_activations.max(1, keepdim=True)[1].type(torch.LongTensor)
            correct += pred.eq(label.view_as(pred)).cpu().sum().item()

            # Classification report
            labels = range(config.num_classes)
            recall += recall_score(label.view(-1), pred.view(-1), labels=labels, average=None)
            precision += precision_score(label.view(-1), pred.view(-1), labels=labels, average=None)
            f1score += f1_score(label.view(-1), pred.view(-1), labels=labels, average=None)

            formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))

            # Print losses
            if batch_idx % config.print_every == 0:
                if config.reconstruction != "None":
                        logging.info(
                            '\nEpoch: {}    Loss: {:.6f}   Margin loss: {:.6f}   Recons. loss: {:.6f}'.format(
                                formatted_epoch,
                                c_loss.item(),
                                m_loss.item(),
                                r_loss.item()))
                else:
                    tepoch.set_postfix(loss=c_loss.item())
    # Print time elapsed for every epoch
    end_time = time.time()
    logging.info('\nEpoch {} takes {:.0f} seconds for training.'.format(formatted_epoch, end_time - start_time))

    # Log train losses
    loss /= len(train_loader)

    if config.reconstruction != "None":
        margin_loss /= len(train_loader)
        recons_loss /= len(train_loader)

    acc = correct / tot_samples

    # Log classification report
    recall /= len(train_loader)
    precision /= len(train_loader)
    f1score /= len(train_loader)

    # Print classification report
    logging.info("Training classification report:")
    for i in range(config.num_classes):
        logging.info(
            "Class: {} Recall: {:.4f} Precision: {:.4f} F1-Score: {:.4f}".format(i, recall[i], precision[i],
                                                                                 f1score[i]))

    # Log losses
    if writer: 
        writer.add_scalar('train/loss', c_loss.item(), epoch)
        if config.reconstruction != "None":
            writer.add_scalar('train/margin_loss', m_loss.item(), epoch)
            writer.add_scalar('train/reconstruction_loss', r_loss.item(), epoch)
        writer.add_scalar('train/accuracy', acc, epoch)

    # Print losses
    if config.reconstruction != "None":
        logging.info("Training loss: {:.4f} Margin loss: {:.4f} Recons loss: {:.4f}".format(loss,
                                                                                            margin_loss,
                                                                                            recons_loss))
    else:
        logging.info("Training loss: {:.4f} ".format(loss))

    logging.info("Training accuracy: {}/{} ({:.2f}%)".format(correct, len(train_loader.sampler),
                                                             100. * correct / tot_samples))
    logging.info(
        "Training error: {}/{} ({:.2f}%)".format(tot_samples - correct, tot_samples,
                                                 100. * (1 - correct / tot_samples)))


    if (config.decay_steps > 0 and epoch % config.decay_steps == 0):
        # Update learning rate
        if (scheduler.get_last_lr()[0] > config.min_lr):
            scheduler.step()
            logging.info('New learning rate: {}'.format(scheduler.get_last_lr()[0]))

    return loss, acc