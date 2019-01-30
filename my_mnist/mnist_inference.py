import argparse
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import cv2

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

def load_graph(frozen_graph):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph", default="optimized_frozen_graph.pb", type=str, help = "Quantized/Frozen model to import")
    args = parser.parse_args()

    graph = load_graph(args.frozen_graph)
    #for op in graph.get_operations():
    #    print(op.name)
    #X = tf.placeholder("float", [None, n_input])
    #Y = tf.placeholder("float", [None, n_classes])
    input_node  = graph.get_tensor_by_name('inputs/X:0')
    output_node = graph.get_tensor_by_name('output/output/BiasAdd:0')
    #mnist = input_data.read_data_sets("/tmp/MNIST_data/data/", one_hot=True)
    #picture = np.ones([1, 784])
    #print('picture:', picture)
    picture = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
    print('picture:', picture)
    picture = picture.reshape(1, 784)
    with tf.Session(graph=graph) as sess:
        #print("Accuracy:", output_node.eval({input_node: mnist.test.images, output: mnist.test.labels}))
        #_ = sess.run(output_node, feed_dict={input_node: mnist.test.images})
        _ = sess.run(output_node, feed_dict={input_node: picture})
        for _output in _:
            print("result:", np.argmax(_output))
