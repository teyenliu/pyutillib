from tensorpack.dataflow import *
import tensorflow as tf
from tqdm import tqdm

ds = LMDBData('/home/liudanny/git/caffe-demos/mnist/mnist_train_lmdb/data.mdb', shuffle=False)
ds = PrefetchData(ds, 1000, 1)
ds = BatchData(ds, 64, use_list=True)
#TestDataSpeed(ds, size=500).start()

k = tf.placeholder(tf.string, shape=(64,))
v = tf.placeholder(tf.string, shape=(64,))

with tf.Session() as sess:
    i = 0
    with tqdm(total=500, leave=True, smoothing=0.2) as pbar:
        for k_data, v_data in ds.get_data():
            sess.run([k, v], feed_dict={k: k_data, v: v_data})
            pbar.update()
            i = i + 1
            if i > 500:
                break
