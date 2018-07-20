import os

import numpy as np
import pickle as cp

from keras.utils import to_categorical

from Codes.sliding_window import sliding_window
from Codes.utils import  mini_batch

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

print("--- Loading data...\n","Data already normalized...")

# X_train, y_train, X_test, y_test = load_dataset('C:/Users/41762/Dropbox/Dataset/Opportunity/wearable_113_ambient_40_gestures_new.data')
X_train, y_train, X_test, y_test = load_dataset('/Users/chentailin/Dropbox/Dataset/Opportunity/wearable_113_ambient_40_gestures_new.data')

print("--- Successfully")


# split the wearable sensor data and ambient sensor data
print("===" * 20)
print("--- Spliting the data into Wearable(F) & Ambient(A)... ")
F_train = X_train[:, :113]
A_train = X_train[:, 113:]

F_test = X_test[:, :113]
A_test = X_test[:, 113:]
print("--- Successfully")
print("===" * 20)

print("{0} actual activities and 1 NaN activity,\nTotal 18 Classes".format(y_train.max()))

# Opportunity num_classes = 18
num_classes = 18

print("F_train shape is ", F_train.shape)
print("F_test shape is ", F_test.shape)

print("A_train shape is ", A_train.shape)
print("A_test shape is ", A_test.shape)


print("y_train shape is ", y_train.shape)
print("y_test shape is ", y_test.shape)

print("===" * 20)



print("--- Changing targets shape... (One-Hot Encoder for cross-entropy loss ) ")
y_train_1hot = to_categorical(y=y_train, num_classes=num_classes)
y_test_1hot = to_categorical(y=y_test, num_classes=num_classes)
print("Before transforming: y_train shape {0}, y_test shape {2} \nAfter transforming: y_train_1hot shape {1},y_test_1hot shape {3}".format(
    y_train.shape, y_train_1hot.shape, y_test.shape, y_test_1hot.shape))

print("--- Successfully")
print("===" * 20)

print("--- Opoorating sliding window...")
WINDOW_LENGTH = 24
WINDOW_STEP =24
F_train = sliding_window(F_train,WINDOW_LENGTH,WINDOW_STEP)
F_test = sliding_window(F_test,WINDOW_LENGTH,WINDOW_STEP)

A_train = sliding_window(A_train,WINDOW_LENGTH,WINDOW_STEP)
A_test = sliding_window(A_test,WINDOW_LENGTH,WINDOW_STEP)

y_train_1hot = sliding_window(y_train_1hot,WINDOW_LENGTH,WINDOW_STEP)
y_test_1hot = sliding_window(y_test_1hot,WINDOW_LENGTH,WINDOW_STEP)

print("Start Training......")

rnn_size = 256
n_classes = y_train_1hot.shape[2]

def lstm_net(x):
    '''

    :param x:
    :param rnn_size:
    :param n_classes:
    :return:
    '''
    weights = {'weights': tf.Variable(tf.random_normal([rnn_size,n_classes]))

    }
    biases = {'biases':tf.Variable(tf.random_normal([n_classes]))

    }
    inputs = tf.unstack(x,x.shape[1],1)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, inputs=inputs ,dtype=tf.float32 )

    output = tf.matmul(outputs,weights['weights']) + biases['biases']

    return output

def train_network(X_train,y_train,X_valid,y_valid,learning_rate = 0.001,num_epochs = 10, batch_size = 128, print_cost=True ):
    '''

    :param X_train:
    :param y_train:
    :param X_valid:
    :param y_valid:
    :param learning_rate:
    :param num_epoch:
    :param batch_size:
    :param print_cost:
    :return:
    '''
    ops.reset_default_graph()
    # initialize the placehodler parameters
    (n_samples, win_len, dim) = X_train.shape
    num_classes = y_train_1hot.shape[2]

    # Create placeholders for X,y and keep_prob
    x = tf.placeholder(dtype=tf.float32, shape=[None, win_len, dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

    # initialize parameters
    init = tf.global_variables_initializer()
    prediction = lstm_net(x)  # prediction
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y))  # loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

    # Run Session
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            if epoch % 3 ==0:
                epoch_loss =0
                minibatches = mini_batch(X_train,y_train, batch_size)
                for minibatch in minibatches:
                    (minibatch_x,minibatch_y) = minibatch

                    _, c = sess.run([optimizer,loss],feed_dict={x:minibatch_x, y:minibatch_y})
                    epoch_loss +=c

                print('Epoch', epoch, 'completed out of',
                      num_epochs, 'Epoch loss', (epoch_loss / batch_size))
                print('Epoch Accuracy:', accuracy.eval({x: X_valid, y: y_valid}))

        print("Optimization Finished!")

batch_size = 128


train_network(F_train,y_train_1hot,F_test,y_test_1hot)










