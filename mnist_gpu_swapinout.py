# Common imports
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import memory_saving_gradients
import mem_util
import linearize as linearize_lib

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

mnist = input_data.read_data_sets("/home/liudanny/mnist_data/")

'''
The purpose is to replace operations' input tensor by our swap in/out operation.
origin_op: the operation that we have swaped out its output tensor
swapin_op: the operation that we use its output tensor to swap in
'''
def tensor_swapin_and_out(g, origin_op, swapin_op):
    added_control = False
    all_ops = g.get_operations()
    #find the origin_op's output tensor name
    origin_op_name = origin_op.values()[0].name

    #search the to_swapin_op which use
    for op in all_ops:
        for i in range(len(op.inputs)):
            if ((op.inputs[i].name == origin_op_name) and
               ("_grad" in op.name)):
                print("op.name:", op.name)
                """
                ('op.name:', u'layer1/L1_SwapOut')
                ('op.name:', u'layer2/MatMul')
                ('op.name:', u'optimizer/gradients/layer1/Sigmoid_grad/SigmoidGrad')
                """
                #Use connect and remap function to reconnect
                ge.connect(ge.sgv(swapin_op), ge.sgv(op).remap_inputs([i]))
                # FIXME:
                # obviously we cannot add more than 1 control dependency for swap_in op
                if added_control is False:
                    added_control = True
                    print("Control Dependency==> swapin_op:", swapin_op, "op:", op)
                    add_control_dependency(all_ops, swapin_op, op)


# find out the target_op's previous operations
def add_control_dependency(all_ops, swapin_op, target_op):
    for tensor in target_op.inputs:
        if "_grad" in tensor.name:
            #we need to find this tenor is which operation's output
            for op in all_ops:
                for i in range(len(op.outputs)):
                    if ((op.outputs[i].name == tensor.name) and
                    ("_grad" in op.name)):
                        print("swapin_op:", swapin_op, "op:", op)
                        ge.add_control_inputs(swapin_op, op)


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

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
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 10

reset_graph()

with tf.device('/cpu:0'):
    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
        y = tf.placeholder(tf.int32, shape=[None], name="y")
        training = tf.placeholder_with_default(False, shape=[], name='training')
    
with tf.device('/gpu:0'):
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv1")
    
    ### Swap in/out ###
    #NOTICE: The last op in conv1 is conv1/Relu
    with tf.device('/cpu:0'):
        conv1_swapout = tf.identity(conv1, name = "Conv1_SwapOut")
        conv1_swapin = tf.identity(conv1_swapout, name = "Conv1_SwapIn")
    
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv2")

    ### Swap in/out ###
    with tf.device('/cpu:0'):
        conv2_swapout = tf.identity(conv2, name = "Conv2_SwapOut")
        conv2_swapin = tf.identity(conv2_swapout, name = "Conv2_SwapIn")
    
    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        ### Swap in/out ###
        with tf.device('/cpu:0'):
            pool3_swapout = tf.identity(pool3, name = "Pool3_SwapOut")
            pool3_swapin = tf.identity(pool3_swapout, name = "Pool3_SwapIn")

        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
        pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)
    
    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
        fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)
    
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc1, n_outputs, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
    
    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
    
with tf.device('/cpu:0'):
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
	
graph = tf.get_default_graph()
graph.as_graph_def()
writer = tf.summary.FileWriter("./simple_graph_events")
writer.add_graph(graph=graph)


### Swap in/out ###
graph = tf.get_default_graph()
origin_op = graph.get_operation_by_name("conv1/Relu")
swapin_op = graph.get_operation_by_name("Conv1_SwapIn")
tensor_swapin_and_out(graph, origin_op, swapin_op)

origin_op = graph.get_operation_by_name("conv2/Relu")
swapin_op = graph.get_operation_by_name("Conv2_SwapIn")
tensor_swapin_and_out(graph, origin_op, swapin_op)

origin_op = graph.get_operation_by_name("pool3/MaxPool")
swapin_op = graph.get_operation_by_name("pool3/Pool3_SwapIn")
tensor_swapin_and_out(graph, origin_op, swapin_op)

### Check the DataFlow Graph ###
graph = tf.get_default_graph()
writer = tf.summary.FileWriter("./simple_graph_events")
writer.add_graph(graph=graph)


n_epochs = 1000
batch_size = 8650

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None 

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.deferred_deletion_bytes=1024
with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True},
                     options=run_options,
                     run_metadata=run_metadata)
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: mnist.validation.images,
                                                y: mnist.validation.labels})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
                mem_use = mem_util.peak_memory(run_metadata)['/gpu:0']/1e6
                print("Memory used: %.2f MB "%(mem_use))

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
                                           y: mnist.validation.labels})
        print("Epoch {}, train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                  epoch, acc_train * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                        y: mnist.test.labels})
    print("Final accuracy on test set:", acc_test)
    save_path = saver.save(sess, "./my_mnist_model")

