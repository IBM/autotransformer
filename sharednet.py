#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:31:24 2023

@author: subhadramokashe
"""
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import scipy
import os
import glob
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class A(nn.Module):
   def __init__(self):
       super(A, self).__init__()
       self.fc = nn.Linear(3, 1)

   def forward(self, x):
       x = self.fc(x)       
       return x

class B(nn.Module):
   def __init__(self, shared_re):
       super(B, self).__init__()
       self.shared_fc = shared_re
       
   def forward(self, x):
       x = self.shared_fc(x)
       return x
       
net_A = A()
net_B = B(shared_re = net_A.fc)
optim_A = optim.Adam(net_A.parameters())
optim_B = optim.Adam(net_B.parameters())
target = torch.randn(1,1)

x_A = torch.rand(1, 3)
y_A_hat = net_A(x_A)
loss_A = F.mse_loss(y_A_hat, target)

x_B = torch.rand(1, 3)
y_B_hat = net_B(x_B)
loss_B = F.mse_loss(y_B_hat, target)

###### A
optim_A.zero_grad()
loss_A.backward()

print(net_A.fc.weight.grad)
#tensor([[0.5219, 0.6192, 0.8145]])
print(net_B.shared_fc.weight.grad)
#tensor([[0.5219, 0.6192, 0.8145]])

optim_A.step()

##### B
optim_B.zero_grad()
loss_B.backward()

print(net_A.fc.weight.grad)
#tensor([[0.6641, 0.0777, 0.2146]])
print(net_B.shared_fc.weight.grad)
# tensor([[0.6641, 0.0777, 0.2146]])

optim_B.step()