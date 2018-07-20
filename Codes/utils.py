import tensorflow as tf
from keras.utils import to_categorical
import numpy as np


def one_hot_encoder(y,num_classes):
    return to_categorical((y,num_classes))


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
    num_bathches = int(x.shape[0] // batch_size)
    for batch_index in range(num_bathches):
        mini_batch_x = x[batch_index *
                         batch_size:((batch_index * batch_size) + batch_size)]
        mini_batch_y = y[batch_index *
                         batch_size:((batch_index * batch_size) + batch_size)]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches


def create_placeholders(dim, win_len, num_classes):
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
    x = tf.placeholder('float32', shape=[None, dim, win_len])
    y = tf.placeholder('float32', shape=[None, num_classes])

    return x, y