'''
Created on Mar 03, 2017

@author: liudanny
'''

# -*- coding: utf-8 -*-
# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.contrib import graph_editor as ge
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
LEARNING_RATE = 0.005    # 0.01 this learning rate will be better! Tested
BATCH_SIZE = 4
DISPLAY_STEP = 1

# Network Parameters
n_samples = data.shape[0]
# Define ESL data input
n_input = 4  
n_final = 2

# tf Graph input
with tf.name_scope('inputs'):
    X = tf.placeholder("float32", [None, n_input], name="ESL_xs")
    Y = tf.placeholder("float32", [None, n_final], name="ESL_ys")

# hidden layer settings
n_hidden_1 = 10
n_hidden_2 = 10


with tf.name_scope('weights'):
    weights = {
        'weights_hl1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],), name='w_hl1'),
        'weights_hl2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],), name='w_hl2'),
        'weights_output': tf.Variable(tf.truncated_normal([n_hidden_2, n_final],), name='w_output'),
    }
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

def add_layers(x):
    with tf.device('/gpu:0'):
        with tf.name_scope('layer1'):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weights_hl1']),
                                   biases['biases_hl1']))
            with tf.device('/cpu:0'):
                layer_1_swapout = tf.identity(layer_1, name = "L1_SwapOut")
                layer_1_swapin = tf.identity(layer_1_swapout, name = "L1_SwapIn")
    
        with tf.name_scope('layer2'):
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['weights_hl2']),
                                   biases['biases_hl2']))
        
        with tf.name_scope('layer_final'):
            layer_final = tf.add(tf.matmul(layer_2, weights['weights_output']),
                                   biases['biases_output'])
    return layer_final


py_x = add_layers(X)
predict_op = py_x


with tf.name_scope('cost'):
    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.squared_difference(py_x, Y))
    tf.summary.scalar('cost', cost)

with tf.device('/gpu:0'):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

class PredictESLService(object):
    def __init__(self, 
                 checkpoint_path, 
                 checkpoint_file,
                 checkpoint_epoch):
        """
        example:
        checkpoint_path = "./ckpt_dir_esl"
        checkpoint_file = "./ckpt_dir_esl/model.ckpt-300.meta"
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file = os.path.join(checkpoint_path, checkpoint_file) + "-" + str(checkpoint_epoch) + ".meta"
        self.sess = None
        #self.inputs = None
        #self.outputs = None

        self.init_session_handler()

    def init_session_handler(self):
        self.sess = tf.Session()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Use the model {}".format(ckpt.model_checkpoint_path))
            saver = tf.train.import_meta_graph(self.checkpoint_file)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

            #self.inputs = json.loads(tf.get_collection('inputs')[0])
            #self.outputs = json.loads(tf.get_collection('outputs')[0])
        else:
            print("No model found, exit now")
            exit()

    def predictESL(self, input_data):
        # Normalization
        teX = scalerX.transform(np.array([input_data]))
        teY = self.sess.run(predict_op, feed_dict={X: teX})
        teY = scalerY.inverse_transform(teY)
        return teY


def trainESL(checkpoint_path,
             checkpoint_file,
             checkpoint_epoch):
    global trX, trY, teX, teY
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    ##### Modify the graph #####
    sess = tf.Session()
    all_ops = sess.graph.get_operations()
    print("all_ops:", all_ops)
    
    #l1_swap_in = sess.graph.get_operation_by_name("layer1/L1_SwapIn")
    #print l1_swap_in.outputs[0]

    #matmul_1_grad = sess.graph.get_operation_by_name("optimizer/gradients/layer2/MatMul_grad/MatMul_1")
    #print matmul_1_grad.inputs[1]
        
    #ret = ge.connect(ge.sgv(l1_swap_in), ge.sgv(matmul_1_grad).remap_inputs([0]))
    ret = ge.connect(ge.sgv("layer1/L1_SwapIn", graph=tf.get_default_graph()), 
                     ge.sgv("optimizer/gradients/layer2/MatMul_grad/MatMul_1", graph=tf.get_default_graph()).remap_inputs([0]))
    sess.close()
    
    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter("./simple_graph_events")
    writer.add_graph(graph=graph)

    ############################

    # Launch the graph
    with tf.Session() as sess:

        merged = tf.summary.merge_all() 
        # tf.train.SummaryWriter soon be deprecated, use following
        writer = tf.summary.FileWriter("logs/", sess.graph)

        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        sess.run(tf.global_variables_initializer())
    
        # restore all variables
        """
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        """
    
        # Training cycle
        for epoch in range(checkpoint_epoch + 1):         
            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]
    
            # Train in batches of 4 inputs
            for start in range(0, n_samples, BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end]})
            
            # Record the summary of weights and biases
            if epoch % 50 == 0:
                rs = sess.run(merged,feed_dict={X: trX, Y: trY})
                writer.add_summary(rs, epoch)
    
            # And print the current accuracy on the training data.
            if epoch % 100 == 0 :
                saver.save(sess, 
                           os.path.join(checkpoint_path, checkpoint_file), 
                           global_step=epoch)
        
            # Loop over all batches
            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch+1),
                "cost=", sess.run(cost, feed_dict={X: trX, Y: trY}))
    
        print("Optimization Finished!")
    
        # USe testing samples to predict the result
        trY_pred = sess.run(predict_op, feed_dict={X: trX})
        teY_pred = sess.run(predict_op, feed_dict={X: teX})
               
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


def main(_):
    #if tf.gfile.Exists(FLAGS.log_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.log_dir)
    #tf.gfile.MakeDirs(FLAGS.log_dir)

    if FLAGS.do_training is True:
        print "To do the training process"
        trainESL("./ckpt_dir_esl", "model.ckpt", FLAGS.epoch)
    else:
        print "To do the prediction process"
        esl_service = PredictESLService("./ckpt_dir_esl", 
                                        "model.ckpt",
                                        FLAGS.epoch)
        ret = esl_service.predictESL([FLAGS.x1, FLAGS.x2, FLAGS.x3, FLAGS.x4])
        print("Result: ", ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--do_training',
        type=bool,
        default=False,
        help='To do the trainig process.'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=0,
        help='The total training epoh.'
    )
    parser.add_argument(
        '--x1',
        type=float,
        default=0,
        help='x1 input variable for prediction.'
    )
    parser.add_argument(
        '--x2',
        type=float,
        default=0,
        help='x2 input variable for prediction.'
    )
    parser.add_argument(
        '--x3',
        type=float,
        default=0,
        help='x3 input variable for prediction.'
    )
    parser.add_argument(
        '--x4',
        type=float,
        default=0,
        help='x4 input variable for prediction.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
"""
('Epoch:', '0501', 'cost=', 0.0058836411)
Optimization Finished!
MSE train: 0.006, test: 0.008
R^2 train: 0.994, test: 0.991
"""
