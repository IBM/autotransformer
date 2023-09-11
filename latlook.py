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
from scipy.linalg import orthogonal_procrustes
"""
with open('ls_acombined1010ep.pkl','rb') as f: visl = pickle.load(f)
with open('atrain_double_data.pkl','rb') as f: datao = pickle.load(f)
with open('atrain_double_labels.pkl','rb') as f: labels = pickle.load(f)

print(visl.shape)
#corrin = np.zeros((10,10))
datas1 = np.zeros((10,28,28))
datas2 = np.zeros((10,28,28))



# for i in range(10):
    

#     idx  = np.argwhere(labels==i)
#     sidx = np.random.choice(np.squeeze(idx), size = 10, replace = False, p = None)   
#     sidx2 = np.setxor1d(sidx, np.squeeze(idx))
#     datas2[i,:,:] = np.mean(datao[sidx2,0,:,:],0)
#     datas1[i,:,:] = np.mean(datao[sidx,0,:,:],0)
#     # datas1[i,:,:] = datao[sidx,0,:,:]
#     # sidx3 = np.random.choice(np.squeeze(idx), size = 1, replace = False, p = None)  
#     # datas2[i,:,:] = datao[sidx3,0,:,:]
    
    
selfsim  = np.zeros((10,10,len(visl)))    

 """   
with open('ls_acombined1ep2.pkl','rb') as f: visl = pickle.load(f)
def sesim(visl)  :  
    selfsim  = np.zeros((10,10,len(visl)))
    for k in range(len(visl)):
        data1 = visl[:,:,:,k]
        argex =  np.setxor1d(np.arange(0,len(visl),dtype = "int"),k)
        argsx = np.random.choice(argex, size = int(len(visl)*0.8), replace = True, p = None) 
        data2 = np.mean(visl[:,:,:,argsx],3)
    
        datas1 = data1.reshape(10,49*100)
        datas2 = data2.reshape(10,49*100)
        #print(datas1.shape, datas2.shape)
        datas1 = datas1/np.linalg.norm(datas1)
        datas2 = datas2/np.linalg.norm(datas2)
        #selfsim[:,:,k] = orthogonal_procrustes(datas1[:,:], datas2[:,:])
        
        for i in range(10):
            for j in range(10):
                c = scipy.stats.pearsonr(datas1[i,:], datas2[j,:])
                #c = orthogonal_procrustes(datas1[i,:], datas2[j,:])
                #print(c)
                selfsim[i,j,k] = c[0]
    return selfsim

selfsim = sesim(visl)

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

decodm, decosd = deco(selfsim)
print(decodm)


plt.imshow(np.mean(selfsim,2), vmin = 0, vmax = 1)
#plt.imshow(np.mean(selfsim,2))#, cmap = "cubehelix")

#plt.axis('off')
plt.ylabel("test digits")
plt.xlabel("train digits")

#plt.xlim(10,20)
#plt.ylim(0,10)
plt.colorbar()
plt.show()
"""
#with open('ls_double1010ep.pkl','rb') as f: visl10 = pickle.load(f)
#with open('ls_adouble1010ep.pkl','rb') as f: audt10 = pickle.load(f)
#with open('ls_combined1010ep.pkl','rb') as f: comb10 = pickle.load(f)
#with open('ls_acombined1010ep.pkl','rb') as f: acomb10 = pickle.load(f)
with open('ls_acombined100.pkl','rb') as f: acomb = pickle.load(f)
with open('ls_combined100.pkl','rb') as f: comb = pickle.load(f)
#with open('ls_adouble100.pkl','rb') as f: audt = pickle.load(f)
#with open('ls_double100.pkl','rb') as f: visl = pickle.load(f)
with open('ls_vis100ep.pkl','rb') as f: vis = pickle.load(f)
with open('ls_aud100ep2.pkl','rb') as f: aud = pickle.load(f)
#print(aud)

selfsimacomb = sesim(acomb)
selfsimcomb = sesim (comb)
selfsimaud = sesim(aud)
selfsimvisl = sesim(vis)
#selfsimacomb10 = sesim(acomb10)
#selfsimcomb10 = sesim (comb10)
#selfsimaud10 = sesim(audt10)
#selfsimvisl10 = sesim(visl10)
labels1 = ["auditory \n autoencoder", "visual \n autoencoder",  "multimodal \n  audio \n autoencoder", "multimodal \n visual \n autoencoder"]

#labels = ["a spectrogram paired \n with 100 spectrograms",  "a spectrogram paired \n with 100 images", "an image paired \n with 100 spectrograms", "an image paried \n with 100 images"]
decodm = np.zeros(4)
decods = np.zeros(4)
decodm[0], decods[0] = deco(selfsimaud)
decodm[1], decods[1] = deco(selfsimvisl)
decodm[2], decods[2] = deco(selfsimacomb)

decodm[3], decods[3] = deco(selfsimcomb)





plt.rcParams["font.size"] = "20"
plt.bar(labels1, decodm, yerr=decods, color = "k", capsize=10)
plt.ylabel("decodability score")
plt.show()


"""