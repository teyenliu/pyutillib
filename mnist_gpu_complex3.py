# To support both python 2 and python 3
#from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"


import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.contrib.memory_stats.python.ops import memory_stats_ops
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_data_flow_ops

batch_size = 85
#batch_size = 5578
n_epochs = 1
height = 200
width = 200
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 10

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

with tf.device('/gpu:0'):
    #h = gen_data_flow_ops.stack_v2(-1, elem_type=dtypes.float32, stack_name="foo")
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")

    conv1_1 = tf.layers.conv2d(conv1, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1_1")
    conv1_2 = tf.layers.conv2d(conv1_1, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1_2")
    conv1_3 = tf.layers.conv2d(conv1_2, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1_3")
    #conv1.op._set_attr('_swap_to_host', attr_value_pb2.AttrValue(i=0))
    conv2 = tf.layers.conv2d(conv1_3, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
    conv2_1 = tf.layers.conv2d(conv2, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2_1")
    conv2_2 = tf.layers.conv2d(conv2_1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2_2")
    conv2_3 = tf.layers.conv2d(conv2_2, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2_3")
    #conv2.op._set_attr('_swap_to_host', attr_value_pb2.AttrValue(i=0))
    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 100 * 100])
        #p1 = gen_data_flow_ops.stack_push_v2(h, pool3_flat, swap_memory=True)
        #p1 = tf.Print(p1, [p1], message='p1 data:')
        #p2 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
        #p2 = tf.Print(p2, [p2], message='p2 data:')
        pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
        #fc1.op._set_attr('_swap_to_host', attr_value_pb2.AttrValue(i=0))
        fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

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


from tensorflow.core.protobuf import rewriter_config_pb2
rewrite_options = rewriter_config_pb2.RewriterConfig(disable_model_pruning=True,
            #constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
            #dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            #layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
            #arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            #min_graph_nodes=-1, 
            memory_optimization=rewriter_config_pb2.RewriterConfig.SWAPPING_HEURISTICS)

graph_options = tf.GraphOptions(rewrite_options=rewrite_options)#, infer_shapes=True)
config = tf.ConfigProto(graph_options=graph_options, allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth=True

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

graph = tf.get_default_graph()
writer = tf.summary.FileWriter("./rewriter_graph1")
writer.add_graph(graph=graph)

import numpy as np
picture = np.ones([batch_size, 200 * 200], dtype=np.float32)
picture_label = np.ones([batch_size], dtype=np.float32)

with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        #for iteration in range(mnist.train.num_examples // batch_size):
        for iteration in range(5):
            #X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: picture, y: picture_label}, options=run_options, run_metadata=run_metadata)
            max_bytes_in_use = sess.run(memory_stats_ops.MaxBytesInUse())/1e6
            print("step:%i, Max Memory used: %.2f MB "%(iteration, max_bytes_in_use))
            """
            for device in run_metadata.step_stats.dev_stats:
                device_name = device.device
                print(".........device:", device.device)
                for node in device.node_stats:
                    print("   ................node_stats:", str(node))
            """
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format(show_dataflow=True, show_memory=True)
            with open('timeline_step_%d.json' % iteration, 'w') as f:
                f.write(chrome_trace)
        #acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        #acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        #print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    #graph = sess.graph
    #writer = tf.summary.FileWriter("./rewriter_graph2")
    #m = tf.train.export_meta_graph(graph=graph)
    #writer.add_graph(graph=graph)

    #save_path = saver.save(sess, "./my_mnist_model")

