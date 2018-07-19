
import tensorflow as tf
import os

import numpy as np
import pandas as pd
import scipy.io  # load the matrix file
import pickle as cp

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Permute, Reshape
from keras import backend as K
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

from tensorflow.contrib import rnn
from tensorflow.python.framework import ops

from sliding_window import sliding_window

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define Loading Data Method
def load_dataset(filename):
    """
    Function to load dataset

    Argument:
        filename : the path of the preprocessed dataset

    Return:

        X_train
        y_train
        X_test
        y_test

    Notice:
        Need use `pickle(python3)` or`cpicckle(python2)` module to load and read dataset.

    """
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: X_train {0}, X_test {1}".format(
        X_train.shape, X_test.shape))
    print(" ..reading instances: y_train {0}, y_test {1}".format(
        y_train.shape, y_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


print("Loading data...")
# X_train, y_train, X_test, y_test = load_dataset('DeepConvLSTM/data/gestures_wearable_ambient.data')
X_train, y_train, X_test, y_test = load_dataset(
    '/Users/chentailin/Dropbox/Dataset/Opportunity/wearable_113_ambient_40_gestures_new.data')
# /Users/chentailin/sharedfolder/Human-Activity-Recognition-CodeHub/opportinity-dataset/DeepConvLSTM/data/gestures_wearable_ambient.data

X_train_origin = X_train.copy()
y_train_origin = y_train.copy()
X_test_origin, y_test_origin = X_test.copy(), y_test.copy()

print("Successfully")
print("---" * 20)

# split the wearable sensor data and ambient sensor data
print("---" * 20)
print("Spliting the data into F & A... ")
X_train_F = X_train[:, :113]
X_train_A = X_train[:, 113:]

X_test_F = X_test[:, :113]
X_test_A = X_test[:, 113:]

# More clearly split
F_train = X_train_F
F_test = X_test_F

A_train = X_train_A
A_test = X_test_A
print("Successfully")

print("---" * 20)
print("{0} actual activities and 1 NaN activity,\nTotal 18 Classes".format(y_train.max()))

# Opportunity num_classes = 18
num_classes = 18

print("Now raw data are prepared as followed(normalized)......")
print("F_train shape is ", F_train.shape)
print("F_test shape is ", F_test.shape)

print("A_train shape is ", A_train.shape)
print("A_test shape is ", A_test.shape)


print("y_train shape is ", y_train.shape)
print("y_test shape is ", y_test.shape)

print("---" * 20)

# Prepare the dataset
win_length = 24 # 2. imgaes are 28*28 you need to go into sequence. So 28 chunks of 28 pixels
sliding_step = 24
dim = 113

def opp_sliding_window(data,win_len,sliding_step):
    return sliding_window(data,win_len,sliding_step)









# Define the model
num_epochs = 10

n_classes = 18
batch_size = 128


rnn_size = 64  # num_hidden_lstm units number

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

learning_rate = 0.001


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):  # a 2*2 size of the window and 2*2 movement of the window step no overlapping
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


def relu(x):
    return tf.nn.relu(x)


def dropout(x, keep_rate):
    return tf.nn.dropout(x, keep_prob=keep_rate)


def permute(x, permute_list):
    return tf.transpose(x, perm=permute_list)


def mini_batch(x, y, batch_size):
    mini_batches = []
    num_bathches = int(np.floor(x.shape[0] / batch_size))
    for batch_index in range(num_bathches):
        mini_batch_x = x[batch_index *
                         batch_size:((batch_index * batch_size) + batch_size), :, :, :]
        mini_batch_y = y[batch_index *
                         batch_size:((batch_index * batch_size) + batch_size), :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches


def create_placeholders(dim, win_len, n_C0, num_classes):
    """
    Arguments:
    dim :           scalar, dimension of the input ()
    win_len :       scalar, window_size of the input()
    n_C0 :          scalar, number of channels of the input(commonly 1,time-series data)
    num_classes :   sclar, number of classes (num_classes(18))

    Return:
    x :  placeholder for the data input, of shape[None, win_length,dim,n_C0] and type = 'float'
    y :  placeholder for the input labels, of shape[None, num_classes] and type = 'float'
    X_train shape is (46495,24,113,1)-->(num_samples,window_length,feature_dims,num_channels)

    Now the x is (46495,113,24,1)
    """
    x = tf.placeholder('float32', shape=[None, dim, win_len, n_C0])
    y = tf.placeholder('float32', shape=[None, num_classes])

    return x, y

    """
    Initialize parameters to be used in this model,  dictionary-like!
    This method is to create variables basically in our model
    did't actually create the neural_network_model
    Process:
    Input --> Conv2D(num_feat_map,kernel_size=(1,5),padding=same) --> MaxPooling2D(pool_size=(1,2),解释[1])-->
    Dropout-->Conv2D(同上                                        ) --> MaxPooling2D (                     )--> Dropout-->Conv2D_Output
          --> LSTM_0 -> LSTM_1 -> Dense/Output(num_classes)

    解释[1]: pool_size为(1,2) 则原始数据的structure = (dim,win_length，channel） 经过pool_size之后dim不变，win_length减半，和channel无关（不对channel进行操作）若pool_size=（2，1），则dim减半（floor除 地板除结果）， win_length不变
    """


def convolutional_lstm_neural_network(x):

        # 5*5 parameters 1 input and 32 feature maps
    cnn_weights = {'W_conv1': tf.Variable(tf.random_normal([1, 5, 1, 16])), 'W_conv2': tf.Variable(tf.random_normal([1, 5, 16, 16]))
                   }
    # biases just the number of the output
    cnn_biases = {'b_conv1': tf.Variable(tf.random_normal([16])),  # 5*5 parameters 1 input and 32 feature maps
                  'b_conv2': tf.Variable(tf.random_normal([16]))
                  # 'b_fc': tf.Variable(tf.random_normal([1024])),
                  # 'out': tf.Variable(tf.random_normal([n_classes])),
                  }

    lstm_weights = {'weights': tf.Variable(tf.random_normal(
        [rnn_size, n_classes]))}  # its a parameters dictionary

    lstm_biases = {'biases': tf.Variable(tf.random_normal([n_classes]))}

    # We get all the parameters then define the model

    """
    Define the data flow chart
    """
    conv1 = conv2d(x, cnn_weights['W_conv1']) + cnn_biases['b_conv1']
    conv1 = relu(conv1)
    conv1 = maxpool2d(conv1)
    conv1 = dropout(conv1, keep_rate)

    conv2 = conv2d(conv1, cnn_weights['W_conv2']) + cnn_biases['b_conv2']
    conv2 = relu(conv2)
    conv2 = maxpool2d(conv2)
    conv2 = dropout(conv2, keep_rate)

    # change the output shape (None, 113, 6, 16) to (None, 6, 113, 16)
    reshape_1 = permute(conv2, [0, 2, 1, 3])
    reshape_2 = tf.reshape(reshape_1, shape=[-1, 6, 113 * 16])

    inputs = tf.unstack(reshape_2, 6, 1)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32)

    output = tf.matmul(
        outputs[-1], lstm_weights['weights']) + lstm_biases['biases']

    return output


def train_neural_network(X_train, y_train, X_valid, y_valid, learning_rate=0.001, num_epochs=6, batch_size=128, print_cost=True):

    ops.reset_default_graph()

    (m, dim, win_len, n_C0) = X_train_cnn.shape
    X_train_cnn.shape
    num_classes = y_train.shape[1]

    # Create placeholders for X,y and keep_prob
    x, y = create_placeholders(dim, win_len, n_C0, num_classes)
    # keep_prob = tf.placeholder(tf.float32)

    # initialize parameters

    prediction = convolutional_lstm_neural_network(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # initialize parameters

    # keep track of accuracies and losses of test and train data set
    # test_losses = []
    # test_accuracies = []
    # train_losses = []
    # train_accuracies = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # step = 1
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_loss = 0
            minibatches = mini_batch(X_train, y_train, batch_size)
            for minibatch in minibatches:
                (minibatch_x, minibatch_y) = minibatch

                _, c = sess.run(
                    [optimizer, cost], feed_dict={x: minibatch_x, y: minibatch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',
                  num_epochs, 'loss', epoch_loss)

            # train_losses.append(epoch_loss)
            # train_accuracies.append(acc)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
        print('Accuracy:', accuracy.eval({x: X_valid, y: y_valid}))

        print("Optimization Finished!")

        # return test_losses, test_accuracies, train_losses, train_accuracies


train_neural_network(X_train_cnn, y_train, X_valid_cnn, y_valid)