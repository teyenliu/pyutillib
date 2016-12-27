'''
Created on Dec 27, 2016

@author: liudanny
'''

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import pandas as pd

# pylint: disable=missing-docstring
import argparse

# Basic model parameters as external flags.
FLAGS = None

# Import MNIST data
data = pd.read_csv("ESL_20161223.csv", header=0).as_matrix()
trX = data[:,0:4]
trY = data[:,4:6]

# Maximum data
max_x1 = np.max(data[:,0])
max_x2 = np.max(data[:,1])
max_x4 = np.max(data[:,3])
max_y1 = np.max(data[:,4])
max_y2 = np.max(data[:,5])

# Normalization
trX[:,0:1] = trX[:,0:1]/max_x1
trX[:,1:2] = trX[:,1:2]/max_x2
trX[:,3:4] = trX[:,3:4]/max_x4
trY[:,0:1] = trY[:,0:1]/max_y1
trY[:,1:2] = trY[:,1:2]/max_y2


# Select From Model
# Meta-transformer for selecting features based on importance weights.
from sklearn.feature_selection import SelectFromModel

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

# Linear Models
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import RidgeCV

# SVM
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import NuSVC

### Method 1 ###
# We use the base estimator MultiTaskLassoCV since the L1 norm promotes sparsity of features.
# MultiTaskLassoCV is for multiple labels.
#clf = MultiTaskLassoCV()
#clf = LassoCV()
clf = MultiTaskElasticNetCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(trX, trY)
new_trX = sfm.transform(trX)
n_features = new_trX.shape[1]
print sfm.get_support(indices=True)


"""
### Method 2 ###
model = SVR(kernel="linear")
# create the RFE model and select 3 attributes
rfe = RFE(model, 3, step=1)
rfe = rfe.fit(trX, trY)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


### Method 3 ###
model = SVR(kernel="linear")
# create the RFE model and select 3 attributes
rfecv = RFECV(model, cv=3, step=1)
new_trX = rfecv.fit(trX, trY)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


### PCA ###
# Feature Extraction
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(trX)
print(pca.explained_variance_)  


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(trX, trY)
# display the relative importance of each attribute
print(model.feature_importances_)
"""
