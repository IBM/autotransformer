#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:02:13 2023

@author: subhadramokashe
"""

import numpy as np
import librosa
import pickle
import cv2
from tqdm import tqdm, trange
import torch
#import torchs
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
from torchvision.datasets.mnist import MNIST
from torchaudio.transforms import Resample, MFCC
import torchaudio
import torchvision
sf = 8000
n_mels = 128
hop_length = 512
n_iter = 32
n_mfcc = 28

train_set = MNIST(root='./../datasets', train=True, download=True, transform=False)
test_set = MNIST(root='./../datasets', train=False, download=True, transform=False)
train_set_array = train_set.data.numpy()
test_set_array = test_set.data.numpy()
train_labels = train_set.targets.numpy()

test_labels = test_set.targets.numpy()
#print(train_set_array.shape)



folders = []
src = "/Users/subhadramokashe/codebook/data/audiomnist/"

for folder in os.listdir(src):
    # only process folders
    if not os.path.isdir(os.path.join(src, folder)):
        continue
    folders.append(folder)
    
#print(folders)

freq_images = []
labels = []
gender = []

class TrimMFCCs: 

	def __call__(self, batch): 
		return batch[:, 1:, :]

class Standardize:

	def __call__(self, batch): 
		for sequence in batch: 
			sequence -= sequence.mean(axis=0)
			sequence /= sequence.std(axis=0)
		return batch 

for folder in sorted(folders):
    #print(folder)
    src2 =  os.path.join(src, folder)
    #print(src2)
    for filepath in sorted(glob.glob(os.path.join(src2, "*.wav"))):
        # infer sample info from name
        dig, vp, rep = filepath.rstrip(".wav").split("/")[-1].split("_")
        #print(dig)
        #y, x = librosa.load(filepath, sr=sr)
        audio, sr = torchaudio.load(filepath)
        audio = Resample(sr, sf)(audio)
        trna = torchvision.transforms.Compose([MFCC(sample_rate = sr, n_mfcc = n_mfcc+1), TrimMFCCs(),Standardize(),])
        mfccs = trna(audio)
        
        if mfccs.shape[2] > 28:
            mfccs = mfccs[:,:,:28]
        elif mfccs.shape[2]<28:
            mfcsz = torch.zeros((1,28,28))
            mfcsz[:,:,0:mfccs.shape[2]] = mfccs
            mfccs = mfcsz
        print(mfccs.shape)
        freq_images.append(mfccs.detach().cpu().numpy())
        labels.append(np.array(int(dig)))
                                        
        
        
plt.imshow(torch.squeeze(mfccs))
plt.show()


print(labels)
image_labels = np.array(labels)
freq_images = np.array(freq_images)
print(freq_images.shape)

idx = np.argwhere(image_labels ==9)

stsi = np.random.choice(np.squeeze(idx), size = 5, replace = False, p = None)
spect5 = freq_images[stsi]
for i in range(5):
    plt.figure(i)
    plt.imshow(np.squeeze(spect5[i]))
plt.show()
with open('mfcc.pkl','wb') as f: pickle.dump(freq_images, f)
with open('labmfcc.pkl','wb') as f: pickle.dump(image_labels, f)



# with open('melspec.pkl','rb') as f2: freq_images= pickle.load(f2)
# with open('labspec.pkl','rb') as f1: image_labels = pickle.load(f1)





    
    
    


    
    
    
    
    
   
