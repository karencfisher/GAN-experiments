import torch
from torch import nn
import numpy as np


def real_loss(discriminator_out: torch.Tensor, smooth: bool=False, gpu: bool=False) -> nn.BCEWithLogitsLoss:
    scalar = 0.9 if smooth else 1
    labels = torch.ones_like(discriminator_out) * scalar
    if gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    return criterion(discriminator_out, labels)

def fake_loss(discriminator_out: torch.Tensor, gpu: bool=False) -> nn.BCEWithLogitsLoss:
    labels = torch.zeros_like(discriminator_out)
    if gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    return criterion(discriminator_out, labels)

def train_discriminator(real_images, discriminator, optimizer, generator,
                        latent_size, gpu=False):
    discriminator.train()
    optimizer.zero_grad()
    
    if gpu:
        real_images = real_images.cuda()
    
    # real images
    real_logits = discriminator(real_images)
    loss = real_loss(real_logits, smooth=True, gpu=gpu)
    
    # fake images
    with torch.no_grad():
        if real_images.shape[1] == 3:
            noise = np.random.uniform(-1, 1, size=(real_images.size(0), latent_size, 1, 1))
        else:
            noise = np.random.uniform(-1, 1, size=(real_images.size(0), latent_size))
        noise = torch.from_numpy(noise).float()
        if gpu:
            noise = noise.cuda()
        fake_images = generator(noise)  
    fake_logits = discriminator(fake_images)
    loss += fake_loss(fake_logits, gpu=gpu)
    
    # backprop
    loss.backward()
    optimizer.step()
    
    return loss


def train_generator(discriminator, generator, optimizer, batch_size, 
                    latent_size, gpu=False, channels=1):
    generator.train()
    optimizer.zero_grad()
    
    # make fake images
    if channels == 3:
        noise = np.random.uniform(-1, 1, size=(batch_size, latent_size, 1, 1))
    else:
        noise = np.random.uniform(-1, 1, size=(batch_size, latent_size))
    noise = torch.from_numpy(noise).float()
    if gpu:
        noise = noise.cuda()
    fake_images = generator(noise)
    
    # run fake images through discriminator and get losses as if real
    fake_logits = discriminator(fake_images)
    loss = real_loss(fake_logits, gpu=gpu)
    
    # backprop
    loss.backward()
    optimizer.step()
    
    return loss

def make_samples(generator, num_samples, latent_size, channels=1, gpu=False):
    generator.eval()
    if channels == 3:
        noise = np.random.uniform(-1, 1, size=(num_samples, latent_size, 1, 1))
    else:
        noise = np.random.uniform(-1, 1, size=(num_samples, latent_size))
    noise = torch.from_numpy(noise).float()
    if gpu:
        noise = noise.cuda()
    return generator(noise)
