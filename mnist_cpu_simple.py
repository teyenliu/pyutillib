# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import tensorflow as tf
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"


n_epochs = 2
batch_size = 10
height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    #fc1_s_g = tf.stop_gradient(fc1)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                     scope="conv[12]|pool3|output")
    print("train_vars", train_vars)
    
    training_op = optimizer.minimize(loss, var_list=train_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


"""
input_binary = False
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format

graph_def = graph_pb2.GraphDef()
with open("graph.pbtxt", "wb") as f:
    if input_binary:
        graph_def.ParseFromString(f.read())
    else:
        text_format.Merge(f.read(), graph_def)    
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/liudanny/MNIST_data/data/")

#TensorFlow SavedModel builder
export_dir = './my_mnist_builder'
if os.path.exists(export_dir):
    import shutil
    shutil.rmtree(export_dir)
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './my_mnist', 'graph.pbtxt')

    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            print("iteration", iteration)

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_mnist/my_mnist_model")

    #TensorFlow SavedModel builder
    mnist_inputs = {'input': tf.saved_model.utils.build_tensor_info(X)}
    mnist_outputs = {'pred_proba': tf.saved_model.utils.build_tensor_info(Y_proba)}
    mnist_signature = tf.saved_model.signature_def_utils.build_signature_def(
        mnist_inputs, mnist_outputs, 'mnist_sig_name')
    builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       {'mnist_signature': mnist_signature})
    builder.save()

