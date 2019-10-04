import argparse
import re
import sys

import onnx
import tensorflow as tf
from tf2onnx import utils

graph_def = tf.GraphDef()
with tf.gfile.FastGFile('frozen_graph.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(graph_def, name='')

ops = tf_graph.get_operations()
#set shape for tensor
#ops[0].outputs[0].set_shape([1,784])


dtypes = {}
output_shapes = {}
shape_override = "inputs/X:0[1,784]"
import tf2onnx.utils
inputs, shape_override = tf2onnx.utils.split_nodename_and_shape(shape_override)

for node in ops:
    for out in node.outputs:
        print(out)
        shape = shape_override.get(out.name)
        print(shape)
        if shape is None:
            try:
                shape = out.get_shape().as_list()
            except Exception as ex:
                shape = []
        dtypes[out.name] = utils.map_tf_dtype(out.dtype)
        output_shapes[out.name] = shape
        print(shape)

print(output_shapes)
