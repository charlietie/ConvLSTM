import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops
# tf.contrib.rnn.MultiRNNCell

from Codes.utils import *

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

def lstm_net(x,rnn_size,n_classes):
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
    inputs = x
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, inputs=inputs ,dtype=tf.float32 )

    output = tf.matmul(outputs,weights['weights']) + biases['biases']

    return output




