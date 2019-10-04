""" To generate the pb file for testing
python /danny/tensorflow/tensorflow/python/tools/freeze_graph.py  \
    --input_graph=./graph.pbtxt \
    --input_checkpoint=./my_mnist_model \
    --output_graph=./frozen_graph.pb \
    --output_node_names=train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits \
    --input_binary=False
"""
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import tensorflow as tf
import os

n_epochs = 1
batch_size = 10

def load_graph(frozen_graph):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
        # get graph nodes
        graph_nodes=[n for n in graph_def.node]
        # get weights
        wts = [n for n in graph_nodes if n.op=='Const']
        # print the wts in details
        from tensorflow.python.framework import tensor_util
        for n in wts:
            print("Name: %s" % n.name, "Data:")
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))

    return graph

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data/data/")

# We retrieve our checkpoint fullpath
model_dir = './my_mnist'
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_dir + "/frozen_graph.pb"

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = False

graph = load_graph(output_graph)
with tf.Session(graph=graph) as sess:

    # We import the meta graph in the current default Graph
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    #saver.restore(sess, input_checkpoint)

    #graph = tf.get_default_graph()
    for op in graph.get_operations():
        print("op name:", op.name, "tensor:", op.values())
        #if op.name == "output/output/BiasAdd":
        #    logits = op
        #if op.name == "train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits":
        #    xentropy = op
    X = graph.get_tensor_by_name("inputs/X:0")
    y = graph.get_tensor_by_name("inputs/y:0")
    logits = graph.get_tensor_by_name("output/output/BiasAdd:0")
    xentropy = graph.get_tensor_by_name("train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0")

    with tf.name_scope("train_again"):
        #xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval_again"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

    for epoch in range(n_epochs):
        #for iteration in range(mnist.train.num_examples // batch_size):
        for iteration in range(100):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            print("iteration", iteration)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

