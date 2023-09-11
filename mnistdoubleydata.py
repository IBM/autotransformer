import numpy as np

import pickle

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



with open('melspec.pkl','rb') as f2: freq_images= pickle.load(f2)
with open('labspec.pkl','rb') as f1: image_labels = pickle.load(f1)




train_combined = np.zeros((25000,2,28,28))
test_combined = np.zeros((5000,2,28,28))
train_combined_labels = np.zeros(25000)
test_combined_labels = np.zeros(5000)

for i in range(0,10):
    idxtsv = np.argwhere(test_labels == i)
    idxtrv = np.argwhere(train_labels == i)
    #print(idx.shape)
    tr1 = np.random.choice(np.squeeze(idxtrv), size = 2500, replace = False, p = None)
    tr2 = np.random.choice(np.squeeze(idxtrv), size = 2500, replace = False, p = None)
    ts1 = np.random.choice(np.squeeze(idxtsv), size = 500, replace = False, p = None)
    ts2 = np.random.choice(np.squeeze(idxtsv), size = 500, replace = False, p = None)
    
    #idxa = np.argwhere(image_labels ==  i)
    #stra = np.random.choice(np.squeeze(idxa), size = 2500, replace = False, p = None)
    #stsa = np.setxor1d(stra, np.squeeze(idxa))
    
    train_combined[i*2500:(i+1)*2500,0,:,:] = train_set_array[tr1]
    train_combined[i*2500:(i+1)*2500,1,:,:] = train_set_array[tr2]
    train_combined_labels[i*2500:(i+1)*2500] = i
    test_combined[i*500: (i+1)*500,0,:,:] = test_set_array[ts1]
    test_combined[i*500: (i+1)*500,1,:,:] = test_set_array[ts2]
    test_combined_labels[i*500:(i+1)*500] = i
    



with open('train_double_data.pkl','wb') as f: pickle.dump(train_combined, f)
with open('train_double_labels.pkl','wb') as f: pickle.dump(train_combined_labels, f)
with open('test_double_data.pkl','wb') as f: pickle.dump(test_combined, f)
with open('test_double_labels.pkl','wb') as f: pickle.dump(test_combined_labels, f)