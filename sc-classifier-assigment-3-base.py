# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'scMarines.pickle'

with open(pickle_file, 'rb') as f:
  loaded = pickle.load(f)
  train_dataset = loaded['train_dataset']
  train_labels = loaded['train_labels']
  valid_dataset = loaded['valid_dataset']
  valid_labels = loaded['valid_labels']
  test_dataset = loaded['test_dataset']
  test_labels = loaded['test_labels']
  del loaded  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


image_size = 64
num_labels = 3
num_hidden_nodes = 16384
beta1 = 0.01
beta2 = 0.01

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size * 3)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = len(train_dataset[0])

batch_size = 32

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size * 3))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  tf_full_train_dataset = tf.constant(train_dataset)

  # Variables.
  weightsL1 = tf.Variable(
    tf.truncated_normal([image_size * image_size * 3, num_hidden_nodes]))
  biasesL1 = tf.Variable(tf.zeros([num_hidden_nodes]))
  regularizer1 = tf.nn.l2_loss(weightsL1)

  weightsL2 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes, num_labels]))
  biasesL2 = tf.Variable(tf.zeros([num_labels]))
  regularizer2 = tf.nn.l2_loss(weightsL2)

  # Training computation.
  logitsL1 = tf.nn.relu(tf.matmul(tf_train_dataset, weightsL1) + biasesL1)
  logitsL2 = tf.matmul(logitsL1, weightsL2) + biasesL2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logitsL2))
  loss = tf.reduce_mean(loss + beta1*regularizer1 + beta2*regularizer2)

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logitsL2)
  valid_prediction = tf.nn.softmax(
    tf.matmul( tf.nn.relu(tf.matmul(tf_valid_dataset, weightsL1) + biasesL1), weightsL2) + biasesL2)
  test_prediction = tf.nn.softmax(
    tf.matmul( tf.nn.relu(tf.matmul(tf_test_dataset, weightsL1) + biasesL1), weightsL2) + biasesL2)
  full_train_prediction = tf.nn.softmax(
    tf.matmul( tf.nn.relu(tf.matmul(tf_full_train_dataset, weightsL1) + biasesL1), weightsL2) + biasesL2)
num_steps = 1500

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  #print("Full train accuracy: %.1f%%" % accuracy(full_train_prediction.eval(), train_labels))
