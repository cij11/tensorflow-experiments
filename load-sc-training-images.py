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
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size, 3),
                             dtype=np.float32)

    num_images = 0
    for image_filename in image_files:
        image_path = os.path.join(folder_path, image_filename)
        print(image_path)
        image_data = load_single_image(image_path)
        if image_data.shape != (image_size, image_size,3):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset[num_images, :, :, :] = image_data
        num_images = num_images + 1

    #Cull dataset size down to those successfully load_image_folder
    dataset = dataset[0:num_images, :, :, :]
    return dataset

def load_single_image(image_path):
    image_data = (ndimage.imread(image_path, False, 'RGB').astype(float) -
              pixel_depth / 2) / pixel_depth
    return image_data


dataset = load_image_folder(image_folder)

def render_image_ascii(image_data):
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

for i in range(25, 45):
    render_image_ascii(dataset[i])
    print(" ")
#print(image_filenames)

image_data = load_single_image(file_path)

print(image_data.shape)
#print(image_data)
