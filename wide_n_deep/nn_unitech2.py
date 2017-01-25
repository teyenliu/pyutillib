'''
Created on Jan 19, 2017

@author: liudanny
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
# pylint: disable=missing-docstring
import argparse

# Basic model parameters as external flags.
FLAGS = None


# Import MNIST data
data = pd.read_csv("view_parameters_yield_mapping2.csv", header=0)
data = data.dropna()
#data["C"] = data["C"].fillna("None")


from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
data['A'] = class_le.fit_transform(data['A'])
data['B'] = class_le.fit_transform(data['B'])
data['G'] = class_le.fit_transform(data['G'])

from sklearn.model_selection import train_test_split
ttl_X, ttl_Y = data.iloc[:, 0:11], data.iloc[:, 11:12]
trX, teX, trY, teY = \
    train_test_split(ttl_X, ttl_Y, test_size=0.2, random_state=0)

# Standardized data
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
trX = stdsc.fit_transform(trX)
teX = stdsc.fit_transform(teX)
#trY = stdsc.fit_transform(trY)
#teY = stdsc.fit_transform(teY)
trY = trY.as_matrix()/100
teY = teY.as_matrix()/100
    

# Visualize encoder setting
# Parameters
LEARNING_RATE = 0.003    # 0.01 this learning rate will be better! Tested
BATCH_SIZE = 20
DISPLAY_STEP = 1

# Network Parameters
n_samples = trX.shape[0]
n_input = 11  # ESL data input
n_final = 1

# tf Graph input (only pictures)
X = tf.placeholder("float32", [None, n_input], "ESL_input_data")
Y = tf.placeholder("float32", [None, n_final], "ESL_output_data")

# hidden layer settings
n_hidden_1 = 44
n_hidden_2 = 44
n_hidden_3 = 44

weights = {
    'weights_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'weights_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'weights_h3': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_3],)),
    'weights_output': tf.Variable(tf.truncated_normal([n_hidden_3, n_final],)),
}

biases = {
    'biases_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'biases_h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'biases_h3': tf.Variable(tf.random_normal([n_hidden_3])),
    'biases_output': tf.Variable(tf.random_normal([n_final])),
}


def add_layers(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weights_h1']),
                                   biases['biases_h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['weights_h2']),
                                   biases['biases_h2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['weights_h3']),
                                   biases['biases_h3']))
    layer_final = tf.add(tf.matmul(layer_3, weights['weights_output']),
                                   biases['biases_output'])
    return layer_final


py_x = add_layers(X)
predict_op = py_x
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(py_x - Y, 2))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(2000 + 1):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of 4 inputs
        for start in range(0, n_samples, BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end]})
    
        # Loop over all batches
        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch+1),
            "cost=", sess.run(cost, feed_dict={X: trX, Y: trY}))

    print("Optimization Finished!")

    #teY = stdsc.inverse_transform(teY)
    pred_result = sess.run(predict_op, feed_dict={X: teX})
    #pred_result = pred_result * teY.std() + teY.mean()
    print ("teY:", teY * 100,
           "pred_result:", pred_result * 100)
    
    # Plot the chart
    plt.figure()
    #plt.subplot(211)
    plt.plot(teY[:], 'bo', pred_result[:], 'k')
    plt.legend(['Unitech testing', 'Predicted prediction'], loc='upper left')
    #plt.subplot(212)
    #plt.plot(trY[:, 1], 'ro', teY[:, 1], 'k')
    #plt.legend(['Original ESL2', 'Predicted ESL2'], loc='upper left')
    
    #plt.plot(trY[:, 0], 'ro', label='Original data1')
    #plt.plot(trY[:, 1], 'bo', label='Original data2')
    #plt.plot(teY[:, 0], 'g-', label='Predicted line1')
    #plt.plot(teY[:, 1], 'y-', label='Predicted line2')
    #plt.legend(['Original ESL1', 'Original ESL2', 'Predicted ESL1', 'Predicted ESL2'], loc='upper left')
    plt.show()
    
