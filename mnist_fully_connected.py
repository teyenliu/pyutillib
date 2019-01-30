import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
from tensorflow.python.grappler import item as gitem
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.core.framework import attr_value_pb2
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Create the model
n_epochs = 1
batch_size=100
x = tf.placeholder(tf.float32, [batch_size, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b
#y.op._set_attr('_swap_to_host', attr_value_pb2.AttrValue(i=0))

# Define loss and optimizer
y_ = tf.placeholder(tf.int64, [batch_size])
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/liudanny/MNIST_data/data")

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './mnist_fully_connected', 'graph.pbtxt') 
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: X_batch, y_: y_batch})
            print("iteration", iteration)

