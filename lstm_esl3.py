
'''
Created on Dec 16, 2017

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


# Hyper Parameters
BATCH_SIZE = 16
TIME_STEP = 4          # rnn time step 
INPUT_SIZE = 1         # rnn input size 
OUTPUT_SIZE =  2
LR = 0.005               # learning rate


# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # shape(batch, 4)
esl = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])
tf_y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])                             # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=16)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    esl,                        # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
outputs = tf.layers.dense(outputs[:, -1, :], 2)              # output based on the last output step
loss = tf.reduce_mean(tf.pow(outputs - tf_y, 2))
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph


n_samples = trX.shape[0]
for epoch in range(300):
    # Shuffle the data before each training iteration.
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]
    
    # Train in batches of 4 inputs
    for start in range(0, n_samples, BATCH_SIZE):
        end = start + BATCH_SIZE
        # use sin predicts cos
        feed_dict = {tf_x: trX[start:end], tf_y: trY[start:end]}
        _, loss_ = sess.run([train_op, loss], feed_dict)     # train

    print("Epoch:", '%04d' % (epoch+1),
          "cost=", sess.run(loss, feed_dict={tf_x: teX, tf_y: teY}))

print("Optimization Finished!")
# USe testing samples to predict the result
trY_pred = sess.run(outputs, feed_dict={tf_x: trX})
teY_pred = sess.run(outputs, feed_dict={tf_x: teX})


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



