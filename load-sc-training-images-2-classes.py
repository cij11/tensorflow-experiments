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

image_size = 32
pixel_depth = 255.0

num_train_per_set = 800
num_valid_per_set = 100
num_test_per_set = 100

data_root = '.'

image_folder_teal = './training-data/teal-marine-wandering-cropped/'
image_folder_dirt = './training-data/badlands-dirt-cropped/'

def load_image_folder(folder_path):
    image_files = os.listdir(folder_path)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                             dtype=np.float32)

    num_images = 0
    for image_filename in image_files:
        image_path = os.path.join(folder_path, image_filename)
        print(image_path)
        image_data = load_single_image(image_path)
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset[num_images, :, :] = image_data
        num_images = num_images + 1

    #Cull dataset size down to those successfully load_image_folder
    dataset = dataset[0:num_images, :, :]
    return dataset

def load_single_image(image_path):
    image_data = (ndimage.imread(image_path, True).astype(float) -
              pixel_depth / 2) / pixel_depth
    return image_data

image_dataset_teal = load_image_folder(image_folder_teal)
image_dataset_dirt = load_image_folder(image_folder_dirt)

np.random.shuffle(image_dataset_teal)
np.random.shuffle(image_dataset_dirt)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

train_dataset, train_labels = make_arrays(2 * num_train_per_set, image_size)
valid_dataset, valid_labels = make_arrays(2 * num_valid_per_set, image_size)
test_dataset, test_labels = make_arrays(2* num_test_per_set, image_size)

def add_labels(label_array, num_classes, num_per_class_per_set):
    label_array[0:num_per_class_per_set] = 0
    label_array[num_per_class_per_set:2*num_per_class_per_set] = 1

add_labels(train_labels, 2, num_train_per_set)
add_labels(valid_labels, 2, num_valid_per_set)
add_labels(test_labels, 2, num_test_per_set)

train_dataset[0:num_train_per_set] = image_dataset_teal[0:num_train_per_set]
train_dataset[num_train_per_set:2*num_train_per_set] = image_dataset_dirt[0:num_train_per_set]

valid_dataset[0:num_valid_per_set] = image_dataset_teal[num_train_per_set:num_train_per_set + num_valid_per_set]
valid_dataset[num_valid_per_set:2*num_valid_per_set] = image_dataset_dirt[num_train_per_set:num_train_per_set + num_valid_per_set]

test_dataset[0:num_test_per_set] = image_dataset_teal[num_train_per_set + num_valid_per_set:num_train_per_set + num_valid_per_set + num_test_per_set]
test_dataset[num_test_per_set:2*num_test_per_set] = image_dataset_dirt[num_train_per_set + num_valid_per_set:num_train_per_set + num_valid_per_set + num_test_per_set]

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


def render_image_ascii(image_data):
    for i in range (0, image_size):
        line = ''
        for j in range (0, image_size):
            if (image_data[i][j] > 0.15):
                line = line + ('/')
            elif (image_data[i][j] < -0.45):
                line = line + '#'
            else:
                line = line + '.'
        print(line)

print("Train sample")
for i in range(0, 10):
    print (train_labels[i])
    render_image_ascii(train_dataset[i])
    print(" ")

print("Valid sample")
for i in range(0, 10):
    print (valid_labels[i])
    render_image_ascii(valid_dataset[i])
    print(" ")

print("Test sample")
for i in range(0, 10):
    print (test_labels[i])
    render_image_ascii(test_dataset[i])
    print(" ")

pickle_file = os.path.join(data_root, 'croppedMarine.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise



statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
