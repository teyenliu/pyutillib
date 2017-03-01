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
    

from tpot import TPOTClassifier, TPOTRegressor
# tpot 
tpot = TPOTRegressor(generations=10, verbosity=2)
tpot.fit(trX, trY)
print(tpot.score(teX, teY))
# 导出
tpot.export('pipeline_yield.py')

#================= use pipeline result ==========================
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, FunctionTransformer, Normalizer
from tpot.operators.preprocessors import ZeroCount

exported_pipeline = make_pipeline(
    ZeroCount(),
    Binarizer(threshold=0.17),
    Normalizer(norm="l1"),
    LassoLarsCV(normalize=True)
)

exported_pipeline.fit(trX, trY)
trY_pred = exported_pipeline.predict(trX)
teY_pred = exported_pipeline.predict(teX)
accuracy = exported_pipeline.score(teX, teY)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(trY, trY_pred),
        mean_squared_error(teY, teY_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(trY, trY_pred),
        r2_score(teY, teY_pred)))

"""
R^2 train: 0.272, test: 0.193
"""