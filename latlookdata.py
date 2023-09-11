#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:06:56 2023

@author: subhadramokashe
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import scipy

#with open('ls_auds.pkl','rb') as f: visl = pickle.load(f)
with open('melspec.pkl','rb') as f: datao = pickle.load(f)
with open('labspec.pkl','rb') as f: labels = pickle.load(f)

print(datao.shape)
print(labels)

#corrin = np.zeros((10,10))
datas1 = np.zeros((10,28,28))
datas2 = np.zeros((10,28,28))
visl = np.zeros((10,3000,28,28))
selfsim  = np.zeros((10,10,2500))   


for i in range(10):
    

    idx  = np.argwhere(labels==i)
    print(idx)
    #sidx = np.random.choice(np.squeeze(idx), size = 1, replace = False, p = None)   
    #sidx2 = np.setxor1d(sidx, np.squeeze(idx))
    #datas2[i,:,:] = datao[sidx2,0,:,:]
    #datas1[i,:,:] = datao[sidx,0,:,:]
    # datas1[i,:,:] = datao[sidx,0,:,:]
    # sidx3 = np.random.choice(np.squeeze(idx), size = 1, replace = False, p = None)  
    # datas2[i,:,:] = datao[sidx3,0,:,:]
    visl[i,:,:,:] = np.squeeze(datao[idx,0,:,:])
    
     

for k in range(2500):
    data1 = visl[:,k,:,:]
    argex =  np.setxor1d(np.arange(0,2500,dtype = "int"),k)
    argsx = np.random.choice(argex, size = int(len(visl)*0.8), replace = True, p = None) 
    data2 = np.mean(visl[:,argsx,:,:],1)
    #plt.imshow(data1[0,:,:])
    #plt.show()
    #plt.imshow(data2[0,:,:])
    #plt.show()

    datas1 = data1.reshape(10,28*28)
    datas2 = data2.reshape(10,28*28)
    #print(datas1.shape, datas2.shape)
    #datas1 = datas1/np.linalg.norm(datas1)
    #datas2 = datas2/np.linalg.norm(datas2)


    
    for i in range(10):
        for j in range(10):
            c = scipy.stats.pearsonr(datas1[i,:], datas2[j,:])
            #print(c)
            selfsim[i,j,k] = c[0]


plt.imshow(np.mean(selfsim,2),vmin = 0, vmax=1)

#plt.axis('off')
plt.ylabel("test digits")
plt.xlabel("train digits")

#plt.xlim(10,20)
#plt.ylim(0,10)
plt.colorbar()
plt.show()
#print(selfsim[0,0], selfsim[11,1])
#cros = selfsim[10:20,0:10]

#for i in range(10):
#    print(selfsim[i,i])

def deco(selfsi):
    diagonal = np.zeros((10,len(selfsi[:,:,0])))
    maxndiagonal = np.zeros((10,len(selfsi[:,:,0])))
    #rdm = np.mean(selfsi,2)
    for k in range(len(selfsi[:,:,0])):
        rdm =selfsi[:,:,k]
        for i in range(10):
            ofd = []
            for j in range(10):
                if j==i:
                    diagonal[i,k] = rdm[i,i]
                else:
                    ofd.append(rdm[i,j])
                    maxndiagonal[i,k] = np.max(ofd)
    #decos = np.where(diagonal> maxndiagonal, 1,0)
    #print(decos.shape)
    #decos = np.mean(decos)
    decos = np.mean(diagonal-maxndiagonal,1)
    decomn = np.mean(decos)
    decosd = np.std(decos)
    return decomn, decosd

selfsimd, sd = deco(selfsim)
print(selfsimd)

"""
for i in range(0,10):
    for j in range(0,10):
        corrin[i,j] = np.dot(data1[i,:],np.transpose(data1[j,:]))
    #plt.plot(data[i,:])
nmax = np.max(corrin)
plt.imshow(corrin/nmax)
plt.xlabel("digit ")
plt.ylabel("digit ")
plt.colorbar()
plt.show()
plt.show()

transform = ToTensor()

test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)
test_set0 = MNIST(root='./../datasets', train=False, download=True, transform=transform)
xx = np.zeros((10,28,28,500))
for i in range(0,10):
    idx = test_set.targets == i
    test_set0.targets = test_set.targets[idx]
    test_set0.data = test_set.data[idx]
    #print(test_set0.shape)
    test_loader = DataLoader(test_set0, shuffle=False)
    j = 0
    for batch in tqdm(test_loader, desc="Testing"):
        x, y = batch
        xx[i,:,:,j] = x
        
corrin2 = np.zeros((10,10))


data2 = np.mean(xx,3)

for i in range(0,10):
    for j in range(0,10):
        corrin2[i,j] = np.dot(data2[i,:,:],np.transpose(data2[j,:,:]))
    #plt.plot(data[i,:])
nmax2 = np.max(corrin2)
plt.imshow(corrin2/nmax2)
plt.xlabel("digit")
plt.ylabel("digit")
plt.colorbar()
plt.show()

"""

