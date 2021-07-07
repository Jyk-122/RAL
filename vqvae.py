import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchkeras
import tensorboardX
from tensorboardX import  SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import numpy as np
from tqdm import tqdm
from sys import path
import time
import os

# from parameters import gen_config

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, num_hiddens, kernel_size=1, stride=1, bias=False),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x


class VQVAE_2D(nn.Module):
    def __init__(self, in_channels,
                       image_size,
                       num_hiddens,
                       compress_factor,
                       num_residual_layers,
                       num_residual_hiddens,
                       num_embeddings,
                       embedding_dim,
                       commitment_cost):
        super().__init__()
        self._in_channels = in_channels
        self._image_size = image_size
        self._fmap_size = image_size // (2 ** compress_factor)
        self._num_hiddens = num_hiddens
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        if compress_factor == 2:
            self.encoder = [ nn.Conv2d(in_channels, num_hiddens // 2, 4, 2, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1),
                             nn.ReLU(inplace=True) ]
        if compress_factor == 3:
            self.encoder = [ nn.Conv2d(in_channels, num_hiddens // 2, 4, 2, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(num_hiddens, num_hiddens, 4, 2, 1),
                             nn.ReLU(inplace=True) ]
        elif compress_factor == 4:
            self.encoder = [ nn.Conv2d(in_channels, num_hiddens // 4, 4, 2, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(num_hiddens // 4, num_hiddens // 2, 4, 2, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(num_hiddens, num_hiddens, 4, 2, 1),
                             nn.ReLU(inplace=True) ]

        self.encoder += [ ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens),
                          nn.Conv2d(num_hiddens, embedding_dim, 3, 1, 1) ]

        self.decoder = [ nn.Conv2d(embedding_dim, num_hiddens, 3, 1, 1),
                         ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)]


        if compress_factor == 2:
            self.decoder += [ nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(num_hiddens // 2, in_channels, 4, 2, 1),
                              nn.ReLU(inplace=True) ]
        elif compress_factor == 3:
            self.decoder += [ nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 2, 4, 2, 1),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(num_hiddens // 2, in_channels, 4, 2, 1),
                              nn.ReLU(inplace=True) ]
        elif compress_factor == 4:
            self.decoder += [ nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 4, 4, 2, 1),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(num_hiddens // 4, num_hiddens // 4, 4, 2, 1),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(num_hiddens // 4, in_channels, 4, 2, 1),
                              nn.ReLU(inplace=True) ]
        
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
    
    def encode(self, x):
        x = self.encoder(x)
        inputs = x.permute(0, 2, 3, 1).contiguous()

        flat_input = inputs.reshape(-1, self._embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        return encoding_indices, inputs
    
    def decode(self, encoding_indices, inputs):
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        out = self.decoder(quantized)
        
        return out

    def forward(self, x):
        x = self.encoder(x)
        inputs = x.permute(0, 2, 3, 1).contiguous()

        flat_input = inputs.reshape(-1, self._embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        out = self.decoder(quantized)

        return out, vq_loss


if __name__ == '__main__':
    model = VQVAE_2D(1, 128, 32, 3, 3, 32, 128, 128, 0.25).cuda()
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            print(name)
    x = torch.randn(1, 1, 128, 128).cuda()
    out, vq_loss = model(x)
    print(out.shape, vq_loss)