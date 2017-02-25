# -*- coding: utf-8 -*-
# coding=utf-8

'''
Created on Feb 25, 2017

@author: liudanny
'''

# -*- coding: big5 -*-
# coding=Big5
from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_parameters_yield_mapping3_new.csv", header=0, encoding = 'big5')
data = data.dropna()

# Split trining data and testing data
feature_num = data.shape[1]
X = data.iloc[:, 0:feature_num -1]
y = data.iloc[:, feature_num - 1:feature_num].as_matrix().ravel()

# Performing one-hot encoding on nominal features
X = pd.get_dummies(X).as_matrix()

# Split trining data and testing data
from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.3, random_state=0)
    

from tpot import TPOTClassifier
# tpot 
tpot = TPOTClassifier(generations=6, verbosity=2)
tpot.fit(trX, trY)
tpot.score(teX, teY)
# 导出
tpot.export('pipeline.py')