#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 23:17:13 2023

@author: subhadramokashe
"""

import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
        #print(patch.shape, patch.flatten().shape)
    return patches


    
def depatchify(patches, n_patches):
    n, np2, sp2 = patches.shape 
    print(patches.shape)
    h = w = int(np.sqrt(np2)*np.sqrt(sp2))
    c = int(np2/(n_patches*n_patches))
    iimages =torch.zeros(n, c, h, w)
    patch_size = int(np.sqrt(sp2))
    for idx, patches in enumerate(patches):
        for i in range(n_patches):
            for j in range(n_patches):
                    #print(patches[i])
                    ipatch = torch.reshape(patches[i * n_patches + j], [1,c,patch_size,patch_size])
                    iimages[idx,:,i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] = ipatch
    return iimages
                        
                                    
    
   
    
    


class MyViT(nn.Module):
  def __init__(self, chw=(1, 28, 28), n_patches=7):
    # Super constructor
    super(MyViT, self).__init__()

    # Attributes
    self.chw = chw # (C, H, W)
    self.n_patches = n_patches

    assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

  def forward(self, images):
    patches = patchify(images, self.n_patches)
    images_i = depatchify(patches, self.n_patches)
    return images_i


if __name__ == '__main__':
  # Current model
  model = MyViT(
    chw=(1, 28, 28),
    n_patches=7
  )

  x = torch.randn(19, 1, 28, 28) # Dummy images
  plt.imshow(x[0,0,:,:])
  plt.show()
  
  patchify2 = torch.nn.Conv2d(1, 8, 16, 16)
  patches2 = patchify2(x)
 

  
  print(model(x).shape) # torch.Size([7, 49, 16])
  
  
# def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
#     """
#     grid_size: int of the grid height and width
#     return:
#     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     grid_h = np.arange(grid_size, dtype=np.float32)
#     grid_w = np.arange(grid_size, dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)

#     grid = grid.reshape([2, 1, grid_size, grid_size])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     if cls_token:
#         pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
#     return pos_embed


# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0

#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

#     emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#     return emb


# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     """
#     embed_dim: output dimension for each position
#     pos: a list of positions to be encoded: size (M,)
#     out: (M, D)
#     """
#     assert embed_dim % 2 == 0
#     omega = np.arange(embed_dim // 2, dtype=np.float)
#     omega /= embed_dim / 2.
#     omega = 1. / 10000**omega  # (D/2,)

#     pos = pos.reshape(-1)  # (M,)
#     out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

#     emb_sin = np.sin(out) # (M, D/2)
#     emb_cos = np.cos(out) # (M, D/2)

#     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb


# decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))



