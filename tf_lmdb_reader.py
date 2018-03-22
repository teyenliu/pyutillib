import os
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables
from tqdm import tqdm

prefix_path = "/home/liudanny/git/caffe-demos/mnist"
batch_size=64

with tf.Session() as sess:
    path = os.path.join(prefix_path, "mnist_train_lmdb", "data.mdb")
    #print(path)
    reader=io_ops.LMDBReader()
    queue = data_flow_ops.FIFOQueue(200, [dtypes.string], shapes=())
    #key, value = reader.read(queue)
    key, value = reader.read_up_to(queue, batch_size)
    queue.enqueue([path]).run()
    queue.close().run()
    with tqdm(total=500, leave=True, smoothing=0.2) as pbar:
        for i in range(1, 500):
            k,v = sess.run([key,value])
            #print(k, v)
            pbar.update()