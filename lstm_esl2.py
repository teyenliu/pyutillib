'''
Created on Dec 15, 2017

@author: liudanny
'''

# -*- coding: utf-8 -*-
# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import sys

# pylint: disable=missing-docstring
import argparse

# Basic model parameters as external flags.
FLAGS = None
LABEL_NUM = 2

# Import MNIST data
data = pd.read_csv("ESL_20161223.csv", header=0, encoding = 'utf-8')
data = data.dropna()

# Split trining data and testing data
feature_num = data.shape[1]
dataX = data.iloc[:, 0:feature_num - LABEL_NUM]
dataY = data.iloc[:, feature_num - LABEL_NUM:feature_num] #.as_matrix().ravel()

# Standardization 
scalerX = preprocessing.StandardScaler().fit(dataX)
dataX = scalerX.transform(dataX)
scalerY = preprocessing.StandardScaler().fit(dataY)
dataY = scalerY.transform(dataY)

# Split trining data and testing data
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

trX, teX, trY, teY = train_test_split(
    dataX, dataY, test_size=0.3, random_state=0)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
BS = 1 	            # batch size
TIME_STEP = 4       # rnn time step
INPUT_SIZE = 1      # rnn input size
OUT_TIME_STEP = 2   # rnn output time step
CELL_SIZE = 32      # rnn cell size
LR = 0.02           # learning rate


# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # shape(batch, 4, 1)
tf_y = tf.placeholder(tf.float32, [None, OUT_TIME_STEP, INPUT_SIZE])                # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
init_s = rnn_cell.zero_state(batch_size=BS, dtype=tf.float32)    # very first hidden state
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                       # input
    initial_state=init_s,       # the initial hidden state
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)

#outputs shape: TensorShape([Dimension(1), Dimension(4), Dimension(32)])
outputs = tf.layers.dense(outputs[:, -1, :], 2)  # shape=(1, 2)
outs = tf.reshape(outputs, [-1, OUT_TIME_STEP, INPUT_SIZE])
#outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
#net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)
#outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])          # reshape back to 3D


loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # initialize var in graph


# Training cycle
n_samples = trX.shape[0]
trX = np.reshape(trX, (-1, BS, TIME_STEP, INPUT_SIZE))
trY = np.reshape(trY, (-1, BS, OUT_TIME_STEP, INPUT_SIZE))
for epoch in range(100):
    # Shuffle the data before each training iteration.
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]
    
    for step in range(n_samples):
        # use sin predicts cos
        if 'final_s_' not in globals():                 # first state, no any hidden state
            feed_dict = {tf_x: trX[step], tf_y: trY[step]}
        else:                                           # has hidden state, so pass it to rnn
            feed_dict = {tf_x: trX[step], tf_y: trY[step], init_s: final_s_}
        _, pred_, final_s_ = sess.run([train_op, loss, final_s], feed_dict)     # train
        # plotting

print("Optimization Finished!")
trY_pred = []
teY_pred = []
#trX = np.reshape(trX, (-1, TIME_STEP, INPUT_SIZE))
teX = np.reshape(teX, (-1, BS, TIME_STEP, INPUT_SIZE))
#trY = np.reshape(trY, (-1, OUT_TIME_STEP, INPUT_SIZE))
# USe testing samples to predict the result
for i in range(trX.shape[0]):
    trY_pred.append(sess.run(outs, feed_dict={tf_x: trX[i]}).flatten())
for i in range(teX.shape[0]):
    teY_pred.append(sess.run(outs, feed_dict={tf_x: teX[i]}).flatten())


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
mean_squared_error(trY, trY_pred),
mean_squared_error(teY, teY_pred)))

print('R^2 train: %.3f, test: %.3f' % (
r2_score(trY, trY_pred),
r2_score(teY, teY_pred)))

# Inverse_transforming for standardization
teY_pred = scalerY.inverse_transform(teY_pred)
teY = scalerY.inverse_transform(teY)

# Plot the chart
plt.figure()
plt.subplot(211)
plt.plot(teY[:, 0], 'bo', teY_pred[:, 0], 'k')
plt.legend(['Original ESL1', 'Predicted ESL1'], loc='upper left')
plt.subplot(212)
plt.plot(teY[:, 1], 'ro', teY_pred[:, 1], 'k')
plt.legend(['Original ESL2', 'Predicted ESL2'], loc='upper left')
plt.show()



