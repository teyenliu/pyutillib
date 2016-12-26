'''
Created on Dec 22, 2016

@author: liudanny
'''

# Fizz Buzz in Tensorflow!
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import tensorflow as tf
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os


rng = np.random

data = pd.read_csv("DOE_Result.csv", header=0).as_matrix()


# let's plot the data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
plt.show()


trX = data[:,0:3]
trY = data[:,7:8]

max_x1 = np.max(data[:,0])
max_x2 = np.max(data[:,1])
max_x3 = np.max(data[:,2])
max_y = np.max(data[:,7])

#Normalization
trX[:,0:1] = trX[:,0:1]/max_x1
trX[:,1:2] = trX[:,1:2]/max_x2
trX[:,2:3] = trX[:,2:3]/max_x3
trY = trY[:,0:1]/max_y

BATCH_SIZE = 3
NUM_PARAM = 3
NUM_EXAMPLES = trX.shape[0]
LEARNING_RATE = 0.08
DISPLAY_STEP = 3

# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

# Our model is a linear regression model
def model(X, w, b):
    return tf.add(tf.matmul(X, w), b)

# Our variables. The input has width NUM_PARAM, and the output has width 1.
X = tf.placeholder("float32", [None, NUM_PARAM])
Y = tf.placeholder("float32", [None, 1])


# Initialize the weights.
w = init_weights([NUM_PARAM, 1])
b = tf.Variable(tf.zeros([1, 1]))

# Predict y given x using the model.
py_x = model(X, w, b)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_sum(tf.pow(py_x - Y, 2)) / (2 * NUM_EXAMPLES)
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = py_x

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

ckpt_dir = "./ckpt_dir_doe"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(40+1):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # And print the current accuracy on the training data.
        if epoch % DISPLAY_STEP == 0 :
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=epoch)
        print("epoch=", epoch, 
              "cost=", sess.run(cost, feed_dict={X: trX, Y: trY}),
               "W=", sess.run(w), 
               "b=", sess.run(b))

    # Do the test
    teX = np.array([[150.0/max_x1, 95.0/max_x2, 500.0/max_x3]])
    #teY = sess.run(predict_op, feed_dict={X: teX})
    result = (teX.dot(sess.run(w)) + sess.run(b)) * 952
    print "w=", sess.run(w), "b=", sess.run(b), "result=", result

    
    teY = np.zeros(trY.shape)
    for indx in range(len(trX)):
        teY[indx, 0] = (trX[indx,:].dot(sess.run(w)) + sess.run(b))
    print trY # Original Label Data
    print teY
    
    # Plot the chart
    plt.plot(trY[:, 0], 'ro', label='Original data')
    plt.plot(teY[:, 0], 'b-', label='Fited line')
    plt.legend(['Original Pull', 'Predicted Pull'], loc='upper left')
    plt.show()