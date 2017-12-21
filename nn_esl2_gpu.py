'''
Created on Dec 20, 2017

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


# Visualize encoder setting
# Parameters
LEARNING_RATE = 0.003    # 0.01 this learning rate will be better! Tested
BATCH_SIZE = 16
DISPLAY_STEP = 1

# Network Parameters
n_samples = data.shape[0]
# Define ESL data input
n_input = 4
n_final = 2


def print_tf_variables(session, vars_collection):
    for var in vars_collection:
        print(var.name, ", ", session.run(var))


# add layers with variables in cpu
def add_layers_cpu_vars(x, collection):
    with tf.name_scope('layer1'):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weights_hl1']),
                                   biases['biases_hl1']))
        #layer_1 = tf.Print(layer_1, [layer_1], 'argmax(layer_1) = ')
    with tf.name_scope('layer2'):
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['weights_hl2']),
                                   biases['biases_hl2']))
        #layer_2 = tf.Print(layer_2, [layer_2], 'argmax(layer_2) = ')
    with tf.name_scope('layer_final'):
        layer_final = tf.add(tf.matmul(layer_2, weights['weights_output']),
                                   biases['biases_output'])

    return layer_final, layer_1, layer_2


# add layers with variables in gpu
def add_layers_gpu_vars(x, collection):
    with tf.name_scope('layer1'):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, gpu_weights['gpu_weights_hl1']),
                                   gpu_biases['gpu_biases_hl1']))
        #layer_1 = tf.Print(layer_1, [layer_1], 'argmax(layer_1) = ')
    with tf.name_scope('layer2'):
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, gpu_weights['gpu_weights_hl2']),
                                   gpu_biases['gpu_biases_hl2']))
        #layer_2 = tf.Print(layer_2, [layer_2], 'argmax(layer_2) = ')
    with tf.name_scope('layer_final'):
        layer_final = tf.add(tf.matmul(layer_2, gpu_weights['gpu_weights_output']),
                                   gpu_biases['gpu_biases_output'])
    with tf.control_dependencies([layer_final]):
        get_l1_output = tf.identity(layer_1)
        get_l2_output = tf.identity(layer_2)

    return layer_final, get_l1_output, get_l2_output


# define cost/loss function
def add_cost(final_output):
    with tf.name_scope('cost'):
        # Define cost/loss,  minimize the squared error
        cost = tf.reduce_mean(tf.pow(final_output - Y, 2))
        tf.summary.scalar('cost', cost)
        return cost


with tf.name_scope('CPU0'):
    vars_collection = []
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # hidden layer settings
        n_hidden_1 = 10
        n_hidden_2 = 10

        # tf Graph input
        with tf.name_scope('inputs'):
            X = tf.placeholder("float32", [None, n_input], name="ESL_xs")
            Y = tf.placeholder("float32", [None, n_final], name="ESL_ys")
    
        with tf.name_scope('weights'):
            weights = {
                'weights_hl1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],), name='w_hl1'),
                'weights_hl2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],), name='w_hl2'),
                'weights_output': tf.Variable(tf.truncated_normal([n_hidden_2, n_final],), name='w_output'),
            }
            vars_collection.append(weights['weights_hl1'])
            vars_collection.append(weights['weights_hl2'])
            vars_collection.append(weights['weights_output'])

            tf.summary.histogram('layer1' + '/weights', weights['weights_hl1'])
            tf.summary.histogram('layer2' + '/weights', weights['weights_hl2'])
            tf.summary.histogram('layer_final' + '/weights', weights['weights_output'])
    
        with tf.name_scope('biases'):
            biases = {
                'biases_hl1': tf.Variable(tf.random_normal([n_hidden_1]), name='b_hl1'),
                'biases_hl2': tf.Variable(tf.random_normal([n_hidden_2]), name='b_hl2'),
                'biases_output': tf.Variable(tf.random_normal([n_final]), name='b_output'),
            }
            tf.summary.histogram('layer1' + '/biases', biases['biases_hl1'])
            tf.summary.histogram('layer2' + '/biases', biases['biases_hl2'])
            tf.summary.histogram('biases_output' + '/biases', biases['biases_output'])
    
        with tf.name_scope('GPU0'):
            with tf.device('/gpu:0'):
                with tf.name_scope('gpu_weights'):
                    gpu_weights = {
                        'gpu_weights_hl1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],), name='gpu_w_hl1'),
                        'gpu_weights_hl2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],), name='gpu_w_hl2'),
                        'gpu_weights_output': tf.Variable(tf.truncated_normal([n_hidden_2, n_final],), name='gpu_w_output'),
                    }
                    
                    # we cannot perform tf.summary.histogram in gpu device
                    #tf.summary.histogram('layer1' + '/gpu_weights', gpu_weights['gpu_weights_hl1'])
                    #tf.summary.histogram('layer2' + '/gpu_weights', gpu_weights['gpu_weights_hl2'])
                    #tf.summary.histogram('layer_final' + '/gpu_weights', gpu_weights['gpu_weights_output'])
                
                with tf.name_scope('gpu_biases'):
                    gpu_biases = {
                        'gpu_biases_hl1': tf.Variable(tf.random_normal([n_hidden_1]), name='gpu_b_hl1'),
                        'gpu_biases_hl2': tf.Variable(tf.random_normal([n_hidden_2]), name='gpu_b_hl2'),
                        'gpu_biases_output': tf.Variable(tf.random_normal([n_final]), name='gpu_b_output'),
                    }

                    # we cannot perform tf.summary.histogram in gpu device
                    #tf.summary.histogram('layer1' + '/gpu_weights', gpu_weights['gpu_weights_hl1'])
                    #tf.summary.histogram('layer1' + '/biases', gpu_biases['gpu_biases_hl1'])
                    #tf.summary.histogram('layer2' + '/biases', gpu_biases['gpu_biases_hl2'])
                    #tf.summary.histogram('biases_output' + '/gpu_biases', gpu_biases['gpu_biases_output'])

                # define layer and model
                py_x, get_l1_output, get_l2_output = add_layers_gpu_vars(X, vars_collection) 
                
                # copy vars
                copy_variable1 = weights['weights_hl1'].assign(gpu_weights['gpu_weights_hl1'])
                copy_variable2 = weights['weights_hl2'].assign(gpu_weights['gpu_weights_hl2'])
                copy_variableo = weights['weights_output'].assign(gpu_weights['gpu_weights_output'])

        # define loss function
        cost = add_cost(py_x)
        
        # define optimizer    
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
   
        train_op = tf.group(optimizer, 
                            copy_variable1, 
                            copy_variable2, 
                            copy_variableo)
 

        # Launch the graph
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            # tf.train.SummaryWriter soon be deprecated, use following
            writer = tf.summary.FileWriter("logs/", sess.graph)
    
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            sess.run(tf.global_variables_initializer())
    
            # Training cycle
            for epoch in range(100 + 1):
                # Shuffle the data before each training iteration.
                p = np.random.permutation(range(len(trX)))
                trX, trY = trX[p], trY[p]
    
                # Train in batches of 4 inputs
                for start in range(0, n_samples, BATCH_SIZE):
                    end = start + BATCH_SIZE
                    _, _l1_out, _l2_out = sess.run([train_op, get_l1_output, get_l2_output], 
                                             feed_dict={X: trX[start:end], Y: trY[start:end]})
                    print "get_l1_output:", _l1_out
                    print "get_l2_output:", _l2_out

                # Record the summary of weights and biases
                if epoch % 50 == 0:
                    rs = sess.run(merged,feed_dict={X: trX, Y: trY})
                    writer.add_summary(rs, epoch)
                    # print out the temp outputs
                    print_tf_variables(sess, vars_collection)
    
                # Loop over all batches
                # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch+1),
                    "cost=", sess.run(cost, feed_dict={X: trX, Y: trY}))
    
print("Optimization Finished!")

