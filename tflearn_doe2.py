from __future__ import print_function

import tflearn
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Training Data
raw_data = pd.read_csv("DOE_Result.csv", header=0).as_matrix()
data = raw_data[:,0:3]
labels = raw_data[:,7]

# Build neural network
net = tflearn.input_data(shape=[None, 3])
net = tflearn.fully_connected(net, 6)
net = tflearn.fully_connected(net, 1, activation='sigmoid')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=1, 
          show_metric=True, snapshot_step=None)

# Predict surviving chances (class 1 results)
pred = model.predict([170, 90, 650])
print("DiCaprio Surviving Rate:", pred[0][0])
