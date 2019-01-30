"""
mnist_gpu_pbtxt2.py
This python code can restore the Meta Graph and check-point 
that mnist_gpu_pbtxt.py does and then re-train the model again.
"""
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import tensorflow as tf
import os

n_epochs = 1
batch_size = 10

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data/data/")

# We retrieve our checkpoint fullpath
model_dir = './my_mnist'
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_dir + "/graph.pbtxt"

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = False

with tf.Session(graph=tf.Graph()) as sess:

    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    graph = tf.get_default_graph()
    for op in graph.get_operations():
        print(op.name)
        if op.name == "train/Adam":
            training_op = op
    X = graph.get_tensor_by_name("inputs/X:0")
    y = graph.get_tensor_by_name("inputs/y:0")
    accuracy = graph.get_tensor_by_name("eval/Mean:0")

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            print("iteration", iteration)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

