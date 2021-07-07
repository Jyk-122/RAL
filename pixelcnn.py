import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends

import torch
import torch.nn as nn


class MaskedConv(nn.Conv2d):
    '''
    Class that implements the masking for both streams vertical and horizontal.
    It is different if it is the first layer (A) or subsequent layers (B)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='A', ver_or_hor='V'):
        assert mask_type in ['A', 'B'], 'only A or B are possible mask types'
        assert ver_or_hor in ['V', 'H'], 'only H or V are possible ver_or_hor types'

        if ver_or_hor == 'H':  # 1XN mask
            pad = (0, (kernel_size - 1) // 2)
            ksz = (1, kernel_size)

        else:  # NxN mask vertical
            ksz = kernel_size
            pad = (kernel_size - 1) // 2

        super().__init__(in_channels, out_channels, kernel_size=ksz, padding=pad)
        self.mask = torch.zeros_like(self.weight)

        if mask_type == 'A':
            if ver_or_hor == 'V':  # NXN mask
                self.mask[:, :, 0:self.mask.shape[2] // 2, :] = 1

            else:  # horizontal 1xN
                self.mask[:, :, :, 0:self.mask.shape[3] // 2] = 1
        else:  # B
            if ver_or_hor == 'V':  # NXN mask
                self.mask[:, :, 0:self.mask.shape[2] // 2, :] = 1
                self.mask[:, :, self.mask.shape[2] // 2, :] = 1

            else:  # horizontal 1xN
                self.mask[:, :, :, 0:self.mask.shape[3] // 2 + 1] = 1

    def __call__(self, x):
        self.weight.data *= self.mask.to(self.weight.device)
        # print(self.weight)
        return super().__call__(x)


class GatedConvLayer(nn.Module):
    '''
    Main building block of the framework. It implements figure 2 of the paper.
    '''
    def __init__(self, in_channels, nfeats, kernel_size=3, mask_type='A'):
        super(GatedConvLayer, self).__init__()
        self.nfeats = nfeats
        self.mask_type = mask_type
        self.vconv = MaskedConv(in_channels=in_channels, out_channels=2 * nfeats, kernel_size=kernel_size,
                                ver_or_hor='V', mask_type=mask_type)

        self.hconv = MaskedConv(in_channels=in_channels, out_channels=2 * nfeats, kernel_size=kernel_size,
                                ver_or_hor='H', mask_type=mask_type)

        self.v_to_h_conv = nn.Conv2d(in_channels=2 * nfeats, out_channels=2 * nfeats, kernel_size=1)  # 1x1 conv

        self.h_to_h_conv = nn.Conv2d(in_channels=nfeats, out_channels=nfeats, kernel_size=1)  # 1x1 conv

    def GatedActivation(self, x):
        return torch.tanh(x[:, :self.nfeats]) * torch.sigmoid(x[:, self.nfeats:])

    def forward(self, x):
        # x should be a list of two elements [v, h]
        iv, ih = x
        ov = self.vconv(iv)
        oh_ = self.hconv(ih)
        v2h = self.v_to_h_conv(ov)
        oh = v2h + oh_

        ov = self.GatedActivation(ov)

        oh = self.GatedActivation(oh)
        oh = self.h_to_h_conv(oh)

        ##############################################################################
        #Due to the residual connection, if we add it from the first layer, ##########
        #the current pixel is included, in my implementation I removed the first #####
        #residual connection to solve this issue #####################################
        ##############################################################################
        if self.mask_type == 'B':
            oh = oh + ih

        return [ov, oh]


class PixelCNN(nn.Module):
    '''
    Class that stacks several GatedConvLayers, the output has Klevel maps.
    Klevels indicates the number of possible values that a pixel can have e.g 2 for binary images or
    256 for gray level imgs.
    '''
    def __init__(self, nlayers, 
                       in_channels, 
                       nfeats, 
                       Klevels=2,
                       embedding_dim=256,
                       ksz_A=5, 
                       ksz_B=3):
        super(PixelCNN, self).__init__()
        self._in_channels = in_channels
        self.embedding = nn.Embedding(Klevels, embedding_dim)
        self.layers = nn.ModuleList(
            [GatedConvLayer(in_channels=in_channels, nfeats=nfeats, mask_type='A', kernel_size=ksz_A)])
        for i in range(nlayers):
            gatedconv = GatedConvLayer(in_channels=nfeats, nfeats=nfeats, mask_type='B', kernel_size=ksz_B)
            self.layers.append(gatedconv)
        #TODO make kernel sizes as params

        self.out_conv = nn.Sequential(
            nn.Conv2d(nfeats, nfeats, 1),
            nn.ReLU(True),
            nn.Conv2d(nfeats, Klevels, 1)
        )

    def forward(self, x):
        # x = x.float()
        x = x.squeeze(1)
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = [x, x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
        logits = self.out_conv(x[1])

        return logits
    
    def gen_latent_code(self, x):
        logits = self.forward(x)
        encoding_indices = torch.max(logits, dim=1, keepdim=True)[1]

        return encoding_indices

    def sample(self, shape, batch_size, device):
        x = torch.zeros(batch_size, 1, shape[0], shape[1]).long().to(device)
        with torch.no_grad():
            for i in range(shape[0]):
                for j in range(shape[1]):
                    logits = self.forward(x)
                    probs = torch.softmax(logits[:, :, i, j], 1)
                    sample = probs.multinomial(1)
                    x[:, :, i, j] = sample
        
        return x.reshape(batch_size, -1, 1).long()


if __name__ == '__main__':
    m = PixelCNN(4, 1, 64, 256, 256, 5, 3).cuda()
    # x = torch.randn(1, 1, 16, 16).cuda()
    x = torch.randint(0, 256, (4, 1, 32, 32)).cuda()
    print(m(x).shape)