import dis
import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################

    # According to LSGAN paper, they used least squares loss as the loss function
    # For discriminator, the a and b in the paper denote the labels for fake data and real data respectively
    # Set a = 0, b = 1, c = b according to the paper
    real_labels = torch.ones(discrim_real.shape[0], 1).cuda()
    fake_labels = torch.zeros(discrim_fake.shape[0], 1).cuda()
    fake_loss = 0.5 * torch.mean((discrim_fake - fake_labels) ** 2)
    real_loss = 0.5 * torch.mean((discrim_real - real_labels) ** 2)
    loss = fake_loss + real_loss
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    gen_labels = torch.ones(discrim_fake.shape[0], 1).cuda()
    loss = 0.5 * torch.mean((discrim_fake - gen_labels) ** 2)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
