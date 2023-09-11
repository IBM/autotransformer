#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:35:00 2023

@author: subhadramokashe
"""
import torch
#import torchs
import torch.nn as nn
pixel_values = torch.randn(1, 3, 112, 112)
embedding = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
print(embedding(pixel_values).shape)