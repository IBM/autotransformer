import numpy as np

import pickle

#from torchvision.datasets.mnist import MNIST
sr = 22050
n_mels = 128
hop_length = 512
n_iter = 32



with open('train_combined_data.pkl','rb') as f: traindata = pickle.load(f)
with open('train_combined_labels.pkl','rb') as f: trainlabels = pickle.load(f)
 
with open('test_combined_data.pkl','rb') as f: testdata = pickle.load(f)
with open('test_combined_labels.pkl','rb') as f: testlabels = pickle.load(f)

print(testlabels.shape)

train_combined = np.zeros((2500000,2,28,28))
test_combined = np.zeros((500000,2,28,28))
train_combined_labels = np.zeros(2500000)
test_combined_labels = np.zeros(500000)

for i in range(0,10):
    idxtrv = np.argwhere(trainlabels == i)
    idxtsv = np.argwhere(testlabels == i)
    #print(idxtrv.shape)
    
    tr1 = np.random.choice(np.squeeze(idxtrv), size = 2500, replace = False, p = None)
    tr1r = np.repeat(tr1, 100)
    #print(tr1r.shape)
    ts1 = np.random.choice(np.squeeze(idxtsv), size = 500, replace = False, p = None)
    ts2 = np.random.choice(np.squeeze(idxtsv), size = 50000, replace = True, p = None)
    ts1r = np.repeat(ts1, 100)
    tr2 = np.random.choice(np.squeeze(idxtrv), size = 250000, replace = True, p = None)
    ttd = traindata[tr1r,0,:,:] 
    #print(ttd.shape)
    
    train_combined[i*250000:(i+1)*250000,0,:,:] = traindata[tr1r,1,:,:]
    train_combined[i*250000:(i+1)*250000,1,:,:] = traindata[tr2,0,:,:]
    train_combined_labels[i*250000:(i+1)*250000] = i
    test_combined[i*50000: (i+1)*50000,0,:,:] = testdata[ts1r,1,:,:]
    test_combined[i*50000: (i+1)*50000,1,:,:] = testdata[ts2,0,:,:]
    test_combined_labels[i*50000:(i+1)*50000] = i
    
    """
    #ts2 = np.random.choice(np.squeeze(idxtsv), size = 500, replace = False, p = None)
    
    #idxa = np.argwhere(image_labels ==  i)
    #stra = np.random.choice(np.squeeze(idxa), size = 2500, replace = False, p = None)
    #stsa = np.setxor1d(stra, np.squeeze(idxa))
    trains = traindata[tr1,0,:,:]
    tests = testdata[ts1,0,:,:]
    
    for j in range(len(tr1)):
    
        train_combined[i*2500+ (j*10) :i*2500+ (j+1)*10 ,0,:,:] = trains[j]
        #print(trains[j].shape)
        
        tr2 = np.random.choice(np.squeeze(idxtrv), size = 10, replace = False, p = None)
        
       
        print(i,j,train_combined[i*2500+ (j*10) :i*2500 + (j+1)*10 ,1,:,:].shape)
        train_combined[i*2500+ (j*10) :i*2500 + (j+1)*10 ,1,:,:] = traindata[tr2,0,:,:]
        #train_combined[i*2500:(i+1)*2500,1,:,:] = train_set_array[tr2]
        train_combined_labels[i*2500+ (j*10) :i*2500+ (j+1)*10] = i
        
    for k in range(len(ts1)-1):
        test_combined[i*500+ (k*10) :i*500+ (k+1)*10 ,0,:,:] = tests[k]
        
        ts2 = np.random.choice(np.squeeze(idxtsv), size = 10, replace = False, p = None)
        test_combined[i*500+ (k*10) :i*500+ (k+1)*10 ,1,:,:] = testdata[ts2,0,:,:]
        #train_combined[i*2500:(i+1)*2500,1,:,:] = train_set_array[tr2]
        test_combined_labels[i*500+ (j*10) :i*500+ (j+1)*10] = i
        
    
"""
print(test_combined_labels)

for j in range(0,10):
    idx0 = test_combined_labels == j
    print(j,idx0)


with open('atrain_combined100_data.pkl','wb') as f: pickle.dump(train_combined, f)
with open('atrain_combined100_labels.pkl','wb') as f: pickle.dump(train_combined_labels, f)
with open('atest_combined100_data.pkl','wb') as f: pickle.dump(test_combined, f)
with open('atest_combined100_labels.pkl','wb') as f: pickle.dump(test_combined_labels, f)
