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
sr = 8000
n_mels = 128
hop_length = 512
n_iter = 32

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



for folder in sorted(folders):
    #print(folder)
    src2 =  os.path.join(src, folder)
    #print(src2)
    for filepath in sorted(glob.glob(os.path.join(src2, "*.wav"))):
        # infer sample info from name
        dig, vp, rep = filepath.rstrip(".wav").split("/")[-1].split("_")
        #print(dig)
        y, x = librosa.load(filepath, sr=sr)
        #y, x = audio, sr = torchaudio.load(filepath)
        print(x)
        S = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=hop_length*2))
        mel_spec = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels, hop_length=hop_length)
        
        img=cv2.resize(np.array(mel_spec),dsize=(28,28))
        #X=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X = img.reshape(1,28,28)

        freq_images.append(X)
        labels.append(np.array(int(dig)))
        # #print(int(dig))
        

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
# with open('melspec.pkl','wb') as f: pickle.dump(freq_images, f)
# with open('labspec.pkl','wb') as f: pickle.dump(labels, f)



# with open('melspec.pkl','rb') as f2: freq_images= pickle.load(f2)
# with open('labspec.pkl','rb') as f1: image_labels = pickle.load(f1)





    
    
    


    
    
    
    
    
   
