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

image_size = 64
pixel_depth = 255.0

image_folder = './training-data/teal-marine-wandering/'
image_file = 'desktop_055353_as.png'

file_path = os.path.join(image_folder, image_file)

print(file_path)



def load_image_folder(folder_path):
    image_files = os.listdir(folder_path)
    return image_files

def load_single_image(image_path):
    image_data = (ndimage.imread(file_path, False, 'RGB').astype(float) -
              pixel_depth / 2) / pixel_depth
    return image_data


image_filenames = load_image_folder(image_folder)

print(image_filenames)

image_data = load_single_image(file_path)

print(image_data.shape)
print(image_data)

for i in range (0, 64):
    line = ''
    for j in range (0, 64):
        if (image_data[i][j][1] > 0.15):
            line = line + ('/')
        elif (image_data[i][j][1] < -0.45):
            line = line + '#'
        else:
            line = line + '.'
    print(line)
