"""
PyTorch implementation of Capsule Networks

Dynamic Routing Between Capsules: https://arxiv.org/abs/1710.09829

Author: Riccardo Renzulli
University: Università degli Studi di Torino, Department of Computer Science

In the documentation I use the following notation to describe tensors shapes

b: batch size
B: number of input capsule types
C: number of output capsule types
ih: input height
iw: input width
oh: output height
ow: output width
is0: first dimension of input capsules
is1: second dimension of input capsules
os0: first dimension of output capsules
os1: second dimension of output capsules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):

    def __init__(self, batch_averaged=True, margin_loss_lambda=0.5, m_plus=0.9, m_minus=0.1, device="cpu"):
        """
        Batch margin loss for class existence.

        :param batch_averaged: Should the losses be averaged (True) or summed (False) over observations
                               for each minibatch.
        :param margin_loss_lambda: Hyperparameter for down-weighting the loss for missing classes.
        :param m_plus: Vector capsule length threshold for correct class.
        :param m_minus: Vector capsule length threshold for incorrect class.
        :param device: cpu or gpu tensor.
        """
        super(MarginLoss, self).__init__()
        self.batch_averaged = batch_averaged
        self.margin_loss_lambda = margin_loss_lambda
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.device = device

    def forward(self, class_caps_activations, targets):
        """
        The class batch margin loss is defined as:

                Eq. (4): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

        where T_k = 1 iff a class k is present.
        The lambda down-weighting the loss for absent classes stops the initial learning from shrinking the lengths
        of the activity vectors of all the class capsules.
        The batch margin loss is simply the sum of the losses of all class capsules.

        :param class_caps_activations: The capsule activations of the last capsule layer, shape [b, B].
        :param targets: One-hot encoded labels tensor, shape [b, B].

        :return: The margin loss (scalar).
        """
        t_k = targets.type(torch.FloatTensor)
        if targets.ndim == 1:
            t_k = F.one_hot(targets, class_caps_activations.size()[1])
        zeros = torch.zeros(class_caps_activations.size())  # zeros: [b, B]
        # Use GPU if available
        zeros = zeros.to(self.device)
        t_k = t_k.to(self.device)

        margin_loss_correct_classes = t_k * (torch.max(zeros, self.m_plus - class_caps_activations) ** 2)
        margin_loss_incorrect_classes = (1 - t_k) * self.margin_loss_lambda * \
                                        (torch.max(zeros, class_caps_activations - self.m_minus) ** 2)
        margin_loss = margin_loss_correct_classes + margin_loss_incorrect_classes  # margin_loss: [b, B]
        margin_loss = torch.sum(margin_loss, dim=-1)  # margin_loss: [b]

        if self.batch_averaged:
            margin_loss = torch.mean(margin_loss)
        else:
            margin_loss = torch.sum(margin_loss)

        # margin_loss: [1]
        return margin_loss


class ReconstructionLoss(nn.Module):

    def __init__(self, batch_averaged=True):
        """
        The reconstruction loss is used to encourage the class capsules to encode the instantiation parameters of
        the input sample.

        :param: batch_averaged: should the losses be averaged (True) or summed (False) over observations
                                for each minibatch.
        """
        super(ReconstructionLoss, self).__init__()
        self.batch_averaged = batch_averaged

    def forward(self, reconstructed_inputs, original_inputs):
        """
        The reconstruction loss is measured by a squared differences between the reconstruction and the original input.

        :param reconstructed_inputs: Decoder outputs of images, shape [b, channels, ih, iw].
        :param original_inputs: Original samples, shape [b, channels, ih, iw].

        :return: The reconstruction loss.
        """
        batch_size = reconstructed_inputs.size(0)
        reconstructed_inputs = reconstructed_inputs.view(batch_size, -1)
        original_inputs = original_inputs.view(batch_size, -1)
        loss = torch.sum((reconstructed_inputs - original_inputs) ** 2, dim=-1)

        if self.batch_averaged:
            batch_loss = torch.mean(loss)
        else:
            batch_loss = torch.sum(loss)

        # batch_loss: [1]
        return batch_loss


class CapsLoss(nn.Module):

    def __init__(self, caps_loss_type, margin_loss_lambda=0.5, reconstruction_loss_lambda=5e-4, batch_averaged=True,
                 reconstruction=False, m_plus=0.9, m_minus=0.1, m_min=0.2, m_max=0.9, device="cpu", writer=None):
        """
        The total capsule loss is margin or spread loss and reconstruction loss combined.
        The reconstruction loss is scaled down by 5e-4, serving as a regularization method.

        :param caps_loss_type: The encoder loss type 
        :param margin_loss_lambda: Hyperparameter for down-weighting the loss for missing classes.
        :param reconstruction_loss_lambda: The scale down factor for the reconstruction loss.
        :param batch_averaged: should the losses be averaged (True) or summed (False) over observations
                               for each mini-batch.
        :param m_plus: Vector capsule length threshold for correct class.
        :param m_minus: Vector capsule length threshold for incorrect class.
        """
        super(CapsLoss, self).__init__()
        self.caps_loss_type = caps_loss_type
        self.device = device
        self.writer = writer
        self.reconstruction_loss_lambda = reconstruction_loss_lambda

        self.caps_loss = MarginLoss(batch_averaged, margin_loss_lambda, m_plus, m_minus, self.device)

        if reconstruction is not None:
            self.reconstruction_loss = ReconstructionLoss(batch_averaged)

    def forward(self, class_caps_activations, targets, original_inputs=None, reconstructed_inputs=None, step=1):
        """
        :param class_caps_activations: The capsule activations of the last capsule layer, shape [b, B].
        :param targets: One-hot encoded labels tensor, shape [b, B].
        :param original_inputs: Original samples, shape [b, channels, ih, iw].
        :param reconstructed_inputs: Decoder outputs of images, shape [b, channels, ih, iw].
        :param step: The relative step.

        :return: (total capsule loss, margin loss, reconstruction loss)
        """
        caps_loss = self.caps_loss(class_caps_activations, targets)
        if reconstructed_inputs is not None:
            reconstruction_loss = self.reconstruction_loss(reconstructed_inputs, original_inputs)

            tot_loss = (caps_loss + self.reconstruction_loss_lambda * reconstruction_loss)

            return tot_loss, caps_loss, reconstruction_loss
        else:
            return caps_loss
