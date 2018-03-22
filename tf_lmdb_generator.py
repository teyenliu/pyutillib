"""
this file is modified from keras implemention of data process multi-threading,
see https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
"""
import time
import numpy as np
import threading
import multiprocessing
from tensorpack.dataflow import *
from tqdm import tqdm
import tensorflow as tf

try:
    import queue
except ImportError:
    import Queue as queue

LMDBData_PATH = '/home/liudanny/git/caffe-demos/mnist/mnist_train_lmdb/data.mdb'
LMDB_SHUFFLE = False

def data_provide():  
    ds = LMDBData(LMDBData_PATH, shuffle=LMDB_SHUFFLE)
    return ds.get_data()


class GeneratorEnqueuer():
    """
    Builds a queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(
                        target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.
        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)

def generator_with_bs(batch_size=32):
    idxs = []
    dps = []
    for idx, dp in data_provide():
        try:
            idxs.append(idx)
            dps.append(dp)

            if len(idxs) == batch_size:
                yield idxs, dps
                idxs = []
                dps = []
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            continue

def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator_with_bs(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=64, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=1,batch_size=64)
    with tqdm(total=500, leave=True, smoothing=0.2) as pbar:
        i = 0
        for i in range(500):
            start = time.time()
            images, labels =  next(gen)
            end = time.time()
            pbar.update()
            #print end-start
            #print("idx:", images[0])
            #print("lmdb_data:", labels[0])

    
    """
    # FIXME: how to use tf.data API to create LMDB Dataset?

    dataset = tf.data.FixedLengthRecordDataset().from_generator(generator,
                                           output_types=tf.float32, 
                                           output_shapes=[tf.float32])
    iter = dataset.make_one_shot_iterator()
    dp_element = iter.get_next()
    with tf.Session() as sess:
        print(sess.run(dp_element))
    """