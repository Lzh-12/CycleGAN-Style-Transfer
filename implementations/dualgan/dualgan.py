import os
import numpy as np
import itertools
import sys
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.utils import parse_arguments, load_config, print_config


def sample_images(batches_done, real_A, real_B, G_AB, G_BA, dataset_name):
    """Saves a generated sample using current training batch"""
    with torch.no_grad():  # 推理时不计算梯度，节省内存
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)

        # 拼接：真实A | 生成B | 重建A
        row1 = torch.cat((real_A, fake_B, recov_A), -1)
        # 拼接：真实B | 生成A | 重建B
        row2 = torch.cat((real_B, fake_A, recov_B), -1)
        # 垂直拼接两行
        img_sample = torch.cat((row1, row2), -2)

        save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=1, normalize=True)


def compute_gradient_penalty(D, real_samples, fake_samples, FloatTensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    validity = D(interpolates)
    fake = Variable(FloatTensor(np.ones(validity.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(opt):
    os.makedirs("images/%s" % opt['dataset_name'], exist_ok=True)
    os.makedirs("checkpoints/dualgan/%s" % opt['dataset_name'], exist_ok=True)

    img_shape = (opt['channels'], opt['img_size'], opt['img_size'])

    cuda = True if opt['cuda'] and torch.cuda.is_available() else False

    # Loss function
    cycle_loss = torch.nn.L1Loss()

    # Loss weights
    lambda_adv = 1
    lambda_cycle = 10
    lambda_gp = 10

    # Initialize generator and discriminator
    G_AB = Generator()
    G_BA = Generator()
    D_A = Discriminator()
    D_B = Discriminator()

    if cuda:
        G_AB.cuda()
        G_BA.cuda()
        D_A.cuda()
        D_B.cuda()
        cycle_loss.cuda()

    if opt['epoch'] != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("checkpoints/dualgan/%s/G_AB_%d.pth" % (opt['dataset_name'], opt['epoch'])))
        G_BA.load_state_dict(torch.load("checkpoints/dualgan/%s/G_BA_%d.pth" % (opt['dataset_name'], opt['epoch'])))
        D_A.load_state_dict(torch.load("checkpoints/dualgan/%s/D_A_%d.pth" % (opt['dataset_name'], opt['epoch'])))
        D_B.load_state_dict(torch.load("checkpoints/dualgan/%s/D_B_%d.pth" % (opt['dataset_name'], opt['epoch'])))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Configure data loader
    transforms_ = [
        transforms.Resize((opt['img_size'], opt['img_size']), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        ImageDataset("%s/%s" % (opt['dataroot'], opt['dataset_name']), transforms_=transforms_),
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt['n_cpu'],
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt['lr'], betas=(opt['b1'], opt['b2'])
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    prev_time = time.time()
    for epoch in range(opt['n_epochs']):
        for i, batch in enumerate(dataloader):

            # Configure input
            imgs_A = Variable(batch["A"].type(FloatTensor))
            imgs_B = Variable(batch["B"].type(FloatTensor))

            # ----------------------
            #  Train Discriminators
            # ----------------------

            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            # Generate a batch of images
            fake_A = G_BA(imgs_B).detach()
            fake_B = G_AB(imgs_A).detach()

            # ----------
            # Domain A
            # ----------

            # Compute gradient penalty for improved wasserstein training
            gp_A = compute_gradient_penalty(D_A, imgs_A.data, fake_A.data)
            # Adversarial loss
            D_A_loss = -torch.mean(D_A(imgs_A)) + torch.mean(D_A(fake_A)) + opt['lambda_gp'] * gp_A

            # ----------
            # Domain B
            # ----------

            # Compute gradient penalty for improved wasserstein training
            gp_B = compute_gradient_penalty(D_B, imgs_B.data, fake_B.data)
            # Adversarial loss
            D_B_loss = -torch.mean(D_B(imgs_B)) + torch.mean(D_B(fake_B)) + opt['lambda_gp'] * gp_B

            # Total loss
            D_loss = D_A_loss + D_B_loss

            D_loss.backward()
            optimizer_D_A.step()
            optimizer_D_B.step()

            if i % opt['n_critic'] == 0:

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Translate images to opposite domain
                fake_A = G_BA(imgs_B)
                fake_B = G_AB(imgs_A)

                # Reconstruct images
                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)

                # Adversarial loss
                G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
                # Cycle loss
                G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
                # Total loss
                G_loss = opt['lambda_adv'] * G_adv + opt['lambda_cycle'] * G_cycle

                G_loss.backward()
                optimizer_G.step()

                # --------------
                # Log Progress
                # --------------

                # Determine approximate time left
                batches_left = opt['n_epochs'] * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt['n_critic'])
                prev_time = time.time()

                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
                    % (
                        epoch,
                        opt['n_epochs'],
                        i,
                        len(dataloader),
                        D_loss.item(),
                        G_adv.data.item(),
                        G_cycle.item(),
                        time_left,
                    )
                )

            # Check sample interval => save sample if there
            if batches_done % opt['sample_interval'] == 0:
                # sample_images(batches_done)
                sample_images(batches_done, imgs_A, imgs_B, G_AB, G_BA, opt['dataset_name'])

            batches_done += 1

        if opt['checkpoint_interval'] != -1 and epoch % opt['checkpoint_interval'] == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "checkpoints/dualgan/%s/G_AB_%d.pth" % (opt['dataset_name'], epoch))
            torch.save(G_BA.state_dict(), "checkpoints/dualgan/%s/G_BA_%d.pth" % (opt['dataset_name'], epoch))
            torch.save(D_A.state_dict(), "checkpoints/dualgan/%s/D_A_%d.pth" % (opt['dataset_name'], epoch))
            torch.save(D_B.state_dict(), "checkpoints/dualgan/%s/D_B_%d.pth" % (opt['dataset_name'], epoch))
            print(f"Checkpoint saved at epoch {epoch}.")


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    # 加载配置文件
    opt = load_config(args)
    # 打印配置信息
    print_config(opt)
    # 训练模型
    train(opt)
