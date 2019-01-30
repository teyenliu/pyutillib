import argparse
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import cv2

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../my_mnist_builder", type=str, help = "")
    args = parser.parse_args()

    picture = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
    print('picture:', picture)
    picture = picture.reshape(1, 784)
    with tf.Session() as sess:
        signature_key = 'mnist_signature'
        input_key = 'input'
        output_key = 'pred_proba'

        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], args.model_dir)
        signature = meta_graph_def.signature_def

        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)
        #print("Accuracy:", output_node.eval({input_node: mnist.test.images, output: mnist.test.labels}))
        #_ = sess.run(output_node, feed_dict={input_node: mnist.test.images})
        _ = sess.run(y, feed_dict={x: picture})
        for _output in _:
            print(_output)
