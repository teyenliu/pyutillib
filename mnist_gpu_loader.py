"""
mnist_gpu_pbtxt.py
This python code can save meta graph and check-point. 
It also use SaveModelBuilder to save model with signature def.
"""
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import tensorflow as tf
import os

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

n_epochs = 1
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
export_dir = "./my_mnist_builder"
with tf.Session(graph=tf.Graph()) as sess:

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_dir)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            print("iteration", iteration)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        #save_path = saver.save(sess, "./my_mnist/my_mnist_model") 
    
    #TensorFlow SavedModel builder
    #mnist_inputs = {'input': tf.saved_model.utils.build_tensor_info(X)}
    #mnist_outputs = {'pred_proba': tf.saved_model.utils.build_tensor_info(Y_proba)}
    #mnist_signature = tf.saved_model.signature_def_utils.build_signature_def(
    #    mnist_inputs, mnist_outputs, 'mnist_sig_name')
    #builder.add_meta_graph_and_variables(sess,
    #                                   [tf.saved_model.tag_constants.TRAINING],
    #                                   {'mnist_signature': mnist_signature})
    #builder.save()
