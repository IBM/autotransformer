#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:35:28 2023

@author: subhadramokashe
"""
import numpy
import librosa
import soundfile
import cv2
import numpy as np
#from torchvision.datasets.mnist import MNIST
# 109000

# parameters
sr = 109000
n_mels = 128
hop_length = 512
n_iter = 32
n_mfcc = None # can try n_mfcc=20

path = "/Users/subhadramokashe/codebook/data/audiomnist/01/5_01_5.wav"

# load audio and create Mel-spectrogram
y, _ = librosa.load(path, sr=sr)
print(y.shape)
S = numpy.abs(librosa.stft(y))
mel_spec = librosa.feature.melspectrogram(S=S)

print(mel_spec.shape)
mel_spec=cv2.resize(np.array(mel_spec),dsize=(144,144))
mel_spec = mel_spec.reshape(1,144,144)
print(mel_spec.shape)

# Invert mel-spectrogram
S_inv = librosa.feature.inverse.mel_to_stft(np.squeeze(mel_spec))
y_inv = librosa.griffinlim(S_inv)

soundfile.write('horig.wav', y, samplerate=sr)
soundfile.write('hinv.wav', y_inv, samplerate=sr)
