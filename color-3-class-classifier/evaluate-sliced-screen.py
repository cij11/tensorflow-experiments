# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'slicedScreen.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  del save  # hint to help gc free up memory
  print('Test set', test_dataset.shape)

image_size = 32
num_labels = 3
num_channels = 3 # rgb

import numpy as np

def reformat(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset

test_datasets = reformat(test_dataset)

print('Test set', test_dataset.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

stride = 1

graph = tf.Graph()

with graph.as_default():
  tf_test_dataset = tf.constant(test_dataset)

  # Load single images to evaluate here
  tf_single_dataset =  tf.placeholder(
    tf.float32, shape=(1, image_size, image_size, num_channels))

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, stride, stride, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pool + layer1_biases)
    conv = tf.nn.conv2d(pool, layer2_weights, [1, stride, stride, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pool + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

  test_prediction = tf.nn.softmax(model(tf_test_dataset))

def predictToCharacter(prediction):
    if (prediction[0] > prediction[1] and prediction[0] > prediction[2]):
        return 'T'
    elif (prediction[1] > prediction[2]):
        return 'R'
    else:
        return '.'

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  saver.restore(session, "color_cropped_3_model")
  print('Initialized')
  evaluation = test_prediction.eval()

  line = ''
  for i in range (0, len(evaluation)):
     if (i % 24 == 0):
         print (line)
         line = ''
     line = line + predictToCharacter(evaluation[i])
