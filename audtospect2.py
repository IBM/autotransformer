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
sr = 22050
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
        data, x = librosa.load(filepath, sr=sr)
        #print(x)
        data = librosa.resample(y=data.astype(np.float32), orig_sr=x, target_sr=8000, res_type="scipy")
        
        #embedded_data = data
        
        if len(data) > 8000:
            raise ValueError("data length cannot exceed padding length.")
        elif len(data) < 8000:
            embedded_data = np.zeros(8000)
            offset = np.random.randint(low = 0, high = 8000 - len(data))
            embedded_data[offset:offset+len(data)] = data
        elif len(data) == 8000:
            # nothing to do here
            embedded_data = data
            pass
           

        ##### AlexNet #####

        # stft, with seleced parameters, spectrogram will have shape (228,230)
        f, t, Zxx = scipy.signal.stft(embedded_data, 8000, nperseg = 455, noverlap = 420, window='hann')
        print(Zxx.shape)
        # get amplitude
        Zxx = np.abs(Zxx[0:225, 2:-3])
        Zxx = np.atleast_3d(Zxx).transpose(2,0,1)
        # convert to decibel
        Zxx = librosa.amplitude_to_db(Zxx, ref = np.max)
        #img=cv2.resize(np.array(Zxx),dsize=(28,28))
        #X=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        freq_images.append(Zxx)
        labels.append(np.array(int(dig)))

        print(Zxx.shape)
        
plt.imshow(np.squeeze(Zxx))
plt.show()



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
with open('melspec2.pkl','wb') as f: pickle.dump(freq_images, f)
with open('labspec2.pkl','wb') as f: pickle.dump(labels, f)



# with open('melspec.pkl','rb') as f2: freq_images= pickle.load(f2)
# with open('labspec.pkl','rb') as f1: image_labels = pickle.load(f1)





    
    
    


    
    
    
    
    
   
