import argparse
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import cv2

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
n_epochs = 1
batch_size = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../my_mnist_builder", type=str, help = "")
    args = parser.parse_args()

    #for op in graph.get_operations():
    #    print(op.name)
    #X = tf.placeholder("float", [None, n_input])
    #Y = tf.placeholder("float", [None, n_classes])
    #input_node  = graph.get_tensor_by_name('prefix/inputs/X:0')
    #output_node = graph.get_tensor_by_name('prefix/output/output/BiasAdd:0')
    #mnist = input_data.read_data_sets("/tmp/MNIST_data/data/", one_hot=True)
    #picture = np.ones([1, 784])
    #print('picture:', picture)

    #picture = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
    #print('picture:', picture)
    #picture = picture.reshape(1, 784)

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/MNIST_data/data/")
    
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["serve"], args.model_dir)
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)

                t, _ = sess.run(["train/Adam", "output/output/BiasAdd:0"], 
                           feed_dict={"inputs/X:0": X_batch, "inputs/y:0": y_batch})
                #for _output in _:
                #    print(_output)
                print("iteration", iteration)
            acc_train = sess.run("eval/Mean", feed_dict={"inputs/X:0": X_batch, "inputs/y:0": y_batch})
            acc_test = sess.run("eval/Mean", feed_dict={"inputs/X:0": mnist.test.images, "inputs/y:0": mnist.test.labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)


               
