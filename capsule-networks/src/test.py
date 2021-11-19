"""
PyTorch implementation of Capsule Networks

Dynamic Routing Between Capsules: https://arxiv.org/abs/1710.09829

Author: Riccardo Renzulli
University: Università degli Studi di Torino, Department of Computer Science
"""

import torch
import time
import numpy as np
import torchvision.utils as vutils
import torch.nn.functional as F
from tqdm import *
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score


def test(logging, config, loader, model, criterion, writer, epoch, device, imgdir, split="test"):
    if split=="test":
        logging.info("-------------------------------------- Testing epoch {} --------------------------------------".format(epoch))
    else:
        logging.info("-------------------------------------- Validation epoch {} --------------------------------------".format(epoch))
    return test_vector_capsnet(logging, config, loader, model, criterion, writer, epoch, device, imgdir, split)


def test_vector_capsnet(logging, config, loader, model, criterion, writer, epoch, device, imgdir, split):
    loss = 0
    tot_samples = len(loader.sampler)
    
    if config.reconstruction is not None:
        margin_loss = 0
        recons_loss = 0
    precision = np.zeros(config.num_classes)
    recall = np.zeros(config.num_classes)
    f1score = np.zeros(config.num_classes)
    balanced_accuracy = 0

    correct = 0

    step = epoch * len(loader) + len(loader)

    labels = range(config.num_classes)
    all_labels = []
    all_pred = []

    model.eval()

    start_time = time.time()
    batch_index = 0
    for data, target in tqdm(loader):
        # Store the indices for calculating accuracy
        label = target.unsqueeze(0).type(torch.LongTensor)

        batch_size = data.size(0)
        # Transform to one-hot indices: [batch_size, 10]
        target_encoded = F.one_hot(target, config.num_classes)

        # Use GPU if available
        data, target_encoded = data.to(device), target_encoded.to(device)

        # Output predictions
        if config.reconstruction != "None":
            class_caps_poses, class_caps_activations, coupling_coefficients, reconstructions = model(data)
            c_loss, m_loss, r_loss = criterion(class_caps_activations, target_encoded, data, reconstructions)

            margin_loss += m_loss.item()
            recons_loss += r_loss.item()

        else:
            class_caps_poses, class_caps_activations, coupling_coefficients = model(data)
            c_loss = criterion(class_caps_activations, target_encoded)

        loss += c_loss.item()

        # Count correct numbers

        pred = class_caps_activations.max(1, keepdim=True)[1].type(torch.LongTensor)
        correct += pred.eq(label.view_as(pred)).cpu().sum().item()

        # Classification report
        label_flat = label.view(-1)
        pred_flat = pred.view(-1)
        all_labels.append(label_flat)
        all_pred.append(pred_flat)
        recall += recall_score(label_flat, pred_flat, labels=labels, average=None)
        precision += precision_score(label_flat, pred_flat, labels=labels, average=None)
        f1score += f1_score(label_flat, pred_flat, labels=labels, average=None)
        balanced_accuracy += balanced_accuracy_score(label_flat, pred_flat)


        batch_index += 1
    # Print time elapsed for every epoch
    end_time = time.time()
    formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))
    logging.info('\nEpoch {} takes {:.0f} seconds for {}.'.format(formatted_epoch, end_time - start_time, split))

    if config.reconstruction != "None":
        # Visualize reconstructed images of the last batch
        num_img = 8
        if num_img > batch_size:
            num_img = batch_size
        reconstructions = reconstructions[0:num_img].view(num_img, config.input_channels, config.input_height, config.input_width)
        originals = data[0:num_img].view(num_img, config.input_channels, config.input_height, config.input_width)
        images = torch.cat((originals, reconstructions), 0)
        images = vutils.make_grid(images, normalize=False, scale_each=False)
        vutils.save_image(images, imgdir + '/img-epoch_{}.png'.format(formatted_epoch), normalize=False, scale_each=False)

    # Log test losses
    loss /= len(loader)
    if config.reconstruction != "None":
        margin_loss /= len(loader)
        recons_loss /= len(loader)
    acc = correct / tot_samples

    if writer: 
        writer.add_scalar('{}/loss'.format(split), loss, epoch)
        if config.reconstruction != "None":
            writer.add_scalar('{}/margin_loss'.format(split), margin_loss, epoch)
            writer.add_scalar('{}/reconstruction_loss'.format(split), recons_loss, epoch)
        writer.add_scalar('{}/accuracy'.format(split), acc, epoch)

    # Log classification report
    recall /= len(loader)
    precision /= len(loader)
    f1score /= len(loader)
    balanced_accuracy /= len(loader)
    if writer:
        writer.add_scalar('{}/balanced_accuracy'.format(split), balanced_accuracy, epoch)

    # Print classification report
    logging.info("{} classification report:".format(split))
    for i in range(config.num_classes):
        logging.info("Class: {} Recall: {:.4f} Precision: {:.4f} F1-Score: {:.4f}".format(i,
                                                                                          recall[i],
                                                                                          precision[i],
                                                                                          f1score[i]))

    logging.info(confusion_matrix(torch.cat(all_labels), torch.cat(all_pred)))

    # Print test losses
    if config.reconstruction != "None":
        if config.caps_loss == "margin":
            logging.info("{} loss: {:.4f} Margin loss: {:.4f} Recons loss: {:.4f}".format(split, loss, margin_loss, recons_loss))
        else:
            logging.info("{} loss: {:.4f} Spread loss: {:.4f} Recons loss: {:.4f}".format(split, loss, margin_loss, recons_loss))

    else:
        logging.info("{} loss: {:.4f}".format(split, loss))

    logging.info("{} accuracy: {}/{} ({:.2f}%)".format(split, correct, tot_samples,
                                                    100. * correct / tot_samples))
    logging.info("{} balanced accuracy: {:.2f}".format(split, 100*balanced_accuracy))
    logging.info("{} error: {}/{} ({:.2f}%)\n".format(split, tot_samples - correct,  tot_samples,
                                                 100. * (1 - correct / tot_samples)))

    return loss, acc


