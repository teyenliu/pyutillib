'''
Created on Dec 23, 2016

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
data = pd.read_csv("ESL_20161223.csv", header=0).as_matrix()
trX = data[:,0:4]
trY = data[:,4:6]

# Maximum data
max_x1 = np.max(data[:,0])
max_x2 = np.max(data[:,1])
max_x4 = np.max(data[:,3])
max_y1 = np.max(data[:,4])
max_y2 = np.max(data[:,5])

# Normalization
trX[:,0:1] = trX[:,0:1]/max_x1
trX[:,1:2] = trX[:,1:2]/max_x2
trX[:,3:4] = trX[:,3:4]/max_x4
trY[:,0:1] = trY[:,0:1]/max_y1
trY[:,1:2] = trY[:,1:2]/max_y2

# Visualize encoder setting
# Parameters
LEARNING_RATE = 0.003    # 0.01 this learning rate will be better! Tested
BATCH_SIZE = 20
DISPLAY_STEP = 1

# Network Parameters
n_samples = data.shape[0]
n_input = 4  # ESL data input
n_final = 2

# tf Graph input (only pictures)
X = tf.placeholder("float32", [None, n_input], "ESL_input_data")
Y = tf.placeholder("float32", [None, n_final], "ESL_output_data")

# hidden layer settings
n_hidden_1 = 10
n_hidden_2 = 10


weights = {
    'weights_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'weights_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'weights_output': tf.Variable(tf.truncated_normal([n_hidden_2, n_final],)),
}

biases = {
    'biases_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'biases_h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'biases_output': tf.Variable(tf.random_normal([n_final])),
    
}


def add_layers(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weights_h1']),
                                   biases['biases_h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['weights_h2']),
                                   biases['biases_h2']))
    layer_final = tf.add(tf.matmul(layer_2, weights['weights_output']),
                                   biases['biases_output'])
    return layer_final


py_x = add_layers(X)
predict_op = py_x
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(py_x - Y, 2))
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
        input_data[0] = input_data[0] / max_x1
        input_data[1] = input_data[1] / max_x2
        input_data[3] = input_data[3] / max_x4
        
        teX = np.array([input_data])
        
        teY = self.sess.run(predict_op, feed_dict={X: teX})
        teY[0, 0] = teY[0, 0] * max_y1
        teY[0, 1] = teY[0, 1] * max_y2
        return teY


def trainESL(checkpoint_path,
             checkpoint_file,
             checkpoint_epoch):
    global trX, trY
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Launch the graph
    with tf.Session() as sess:
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
    
            # And print the current accuracy on the training data.
            if epoch % 10 == 0 :
                saver.save(sess, 
                           os.path.join(checkpoint_path, checkpoint_file), 
                           global_step=epoch)
        
            # Loop over all batches
            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch+1),
                "cost=", sess.run(cost, feed_dict={X: trX, Y: trY}))
    
        print("Optimization Finished!")
    
        teX = trX[0:1,:]
        teY = trY[0:1,:]
        esl_result = sess.run(predict_op, feed_dict={X: teX})
        print ("teY:", [teY[0][0] * max_y1, teY[0][1] * max_y2],
               "esl_result:", [esl_result[0][0] * max_y1, esl_result[0][1] * max_y2]) 
        
        
        teY = np.zeros(trY.shape)
        for indx in range(n_samples):
            esl_result = sess.run(predict_op, feed_dict={X: trX[indx:indx+1,:]})
            teY[indx, 0] = esl_result[0, 0] #* max_y1
            teY[indx, 1] = esl_result[0, 1] #* max_y2
        
        # Plot the chart
        plt.figure()
        plt.subplot(211)
        plt.plot(trY[:, 0], 'bo', teY[:, 0], 'k')
        plt.legend(['Original ESL1', 'Predicted ESL1'], loc='upper left')
        plt.subplot(212)
        plt.plot(trY[:, 1], 'ro', teY[:, 1], 'k')
        plt.legend(['Original ESL2', 'Predicted ESL2'], loc='upper left')
        
        #plt.plot(trY[:, 0], 'ro', label='Original data1')
        #plt.plot(trY[:, 1], 'bo', label='Original data2')
        #plt.plot(teY[:, 0], 'g-', label='Predicted line1')
        #plt.plot(teY[:, 1], 'y-', label='Predicted line2')
        #plt.legend(['Original ESL1', 'Original ESL2', 'Predicted ESL1', 'Predicted ESL2'], loc='upper left')
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
