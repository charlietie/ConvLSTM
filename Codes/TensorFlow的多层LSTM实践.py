# -*- coding:utf-8 -*-
# %matplotlib inline
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt
import numpy as np
import pylab

rnn_size = 128
keep_prob = tf.placeholder(tf.float32,[])
num_layers =2
num_classes = 18
# rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple()])

# 定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
def lstm_cell():
    lstm_cell = rnn.DropoutWrapper(cell=rnn.BasicLSTMCell(rnn_size),
                               input_keep_prob=1,
                               output_keep_prob=keep_prob,
                               state_keep_prob=keep_prob
                               )
    return lstm_cell


#调用 MultiRNNCell 来实现多层 LSTM
def recurrent_neural_network(x,rnn_tuple_state,keep_prob):
    stacked_lstm = rnn.MultiRNNCell(cells = [lstm_cell() for _ in range(num_layers)])

#用全零来初始化state
    x = tf.transpose(x,[1,0,2])
    outputs,states = tf.nn.dynamic_rnn(stacked_lstm,
                                       inputs=x,
                                       initial_state=,
                                       dtype=tf.float32,
                                       time_major=True
                                       )
    output = tf.layers.dense(input=outputs, units=num_classes)
    output = tf.transpose(output,[1,0,2])

    return output,states

def train_neural_nework(x,y,batch_size):
    print("Batch size is ", batch_size)
    sequence_len = len(x)//batch_size













