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
from tensorflow.python.tools import freeze_graph

fc1_bias_arr = [ 0.04256172, -0.00969896, -0.03594858, -0.02367548,  0.00042766, -0.01795764,
 -0.00687283, -0.00755484, -0.01338008, -0.00826145, -0.01769337,  0.02237733,
 -0.0078018 , -0.0192155 , -0.02865391, -0.01455068,  0.04860795, -0.07098585,
  0.01934407,  0.02362501, -0.0081314 ,  0.08469316, -0.00820399,  0.00883264,
  0.00303113, -0.0364973 , -0.05508992, -0.00600548,  0.03492954, -0.07770814,
 -0.00608333,  0.06516619,  0.02383774, -0.0065165 , -0.00928443, -0.00846744,
 -0.01869485,  0.01257676,  0.02932306, -0.00320015,  0.00249391,  0.01708212,
 -0.00066147,  0.02651091,  0.00656286, -0.00817399, -0.05923214,  0.06711603,
  0.02925586, -0.03621595, -0.03637085,  0.04133036, -0.02492601, -0.01222748,
 -0.0306641 , -0.02270305,  0.06796722, -0.01060008,  0.04593673,  0.02334813,
  0.06454131, -0.06328842, -0.00795438, -0.01552526]


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

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data/data/")

# for Tensorboard
graph = tf.get_default_graph()
writer = tf.summary.FileWriter("./simple_graph_events")
writer.add_graph(graph=graph)

reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
print(reuse_vars_dict)
"""
{u'output/kernel:0': u'output/kernel:0', u'fc1/bias:0': u'fc1/bias:0', 
 u'conv1/bias:0': u'conv1/bias:0', u'conv1/kernel:0': u'conv1/kernel:0', 
 u'conv2/kernel:0': u'conv2/kernel:0', u'fc1/kernel:0': u'fc1/kernel:0', 
 u'conv2/bias:0': u'conv2/bias:0', u'output/bias:0': u'output/bias:0'}
"""

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './my_mnist', 'graph.pbtxt', as_text=True) 
    init.run()

    my_fc1_bias = [v for v in tf.global_variables() if v.name == "fc1/bias:0"][0]
    assign_my_fc1_bias = my_fc1_bias.assign(fc1_bias_arr)
    sess.run(assign_my_fc1_bias)  # or `assign_op.op.run()`
    print(my_fc1_bias.eval())

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if (iteration % batch_size == 0):
                print("iteration", iteration)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path = saver.save(sess, "./my_mnist/my_mnist_model") 

    #print out trainable variables
    for var in reuse_vars:
        if var.name == "fc1/bias:0":
            weights_data = var.eval()
            print("Name:", var.name, "Data:", weights_data)

    # Freeze the graph
    input_graph_path = './my_mnist/graph.pbtxt'
    checkpoint_path = './my_mnist/my_mnist_model'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "output/output/BiasAdd"
    restore_op_name = ""
    filename_tensor_name = ""
    output_frozen_graph_name = './my_mnist/frozen_graph.pb'
    output_optimized_graph_name = './my_mnist/optimized_frozen_graph.pb'
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")

    #TensorFlow SavedModel builder
    export_dir = './my_mnist_builder'
    if os.path.exists(export_dir):
        import shutil
        shutil.rmtree(export_dir)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    mnist_inputs = {'input': tf.saved_model.utils.build_tensor_info(X)}
    mnist_outputs = {'pred_proba': tf.saved_model.utils.build_tensor_info(Y_proba)}
    mnist_signature = tf.saved_model.signature_def_utils.build_signature_def(
        mnist_inputs, mnist_outputs, 'mnist_sig_name')
    builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       {'mnist_signature': mnist_signature})
    builder.save()

    # UFF format
    # freeze graph and remove nodes used for training
    import uff
    graph_def = tf.get_default_graph().as_graph_def()
    print(graph_def)
    model_output = 'output/output/BiasAdd'
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, [model_output])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    # Create UFF model and dump it on disk
    uff_model = uff.from_tensorflow(frozen_graph, [model_output])
    dump = open('my_mnist/MNIST_simple_cnn.uff', 'wb')
    dump.write(uff_model)
    dump.close()

