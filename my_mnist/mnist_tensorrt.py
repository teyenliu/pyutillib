import pycuda.driver as cuda
import pycuda.autoinit
import argparse

import tensorrt as trt
from tensorrt.parsers import uffparser

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import cv2

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# Run inference on device
def infer(context, input_img, batch_size):
    # load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    # create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * batch_size
    # convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype=np.float32)

    # alocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    # return predictions
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--uff_graph", default="MNIST_simple_cnn.uff", type=str, help = "MNIST model to import")
    args = parser.parse_args()

    # prepare parser
    uff_model = open(args.uff_graph, 'rb').read()
    parser = uffparser.create_uff_parser()
    parser.register_input("inputs/X", (1, 28, 28), 0)
    parser.register_output("output/output/BiasAdd")

    # to create trt engine
    trt_logger = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
    engine = trt.utils.uff_to_trt_engine(logger=trt_logger,
                                     stream=uff_model,
                                     parser=parser,
                                     max_batch_size=1, # 1 sample at a time
                                     max_workspace_size= 1 << 30, # 1 GB GPU memory workspace
                                     datatype=trt.infer.DataType.FLOAT) # that's very cool, you can set precision

    context = engine.create_execution_context()

    picture = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
    #print('picture:', picture.shape)
    picture = picture.reshape(1, 28, 28)
    #print('picture:', picture.shape)
    picture = picture / 255.0

    # Use infer helper function
    prediction = infer(context, picture, 1)
    print(prediction, prediction.shape)
    print("The result is: ", np.argmax(prediction))

