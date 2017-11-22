from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


data_root = "."
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

loaded_pickle = 0

try:
  f = open(pickle_file, 'rb')

  loaded_pickle = pickle.load(f)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

trainX = loaded_pickle['train_dataset'][0:2000]
trainY = loaded_pickle['train_labels'][0:2000]
nsamples, nx, ny = trainX.shape
d2_train_dataset = trainX.reshape((nsamples,nx*ny))

logreg = LogisticRegression()
#logreg.fit(d2_train_dataset, trainY)

testX = loaded_pickle['test_dataset']
testY = loaded_pickle['test_labels']
nsamples2, nx2, ny2 = testX.shape
d2_train_dataset2 = testX.reshape((nsamples2,nx2*ny2))

#Z = logreg.predict(d2_train_dataset2)

#for i in range (0, 49):
#    print(i)
#    print(Z[i])
#    print(testY[i])

print('LogisticRegression score: %f'
     % logreg.fit(d2_train_dataset, trainY).score(d2_train_dataset2, testY))
