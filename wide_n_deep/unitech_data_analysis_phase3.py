# -*- coding: utf-8 -*-
# coding=utf-8

'''
Created on Mar 30, 2017

@author: liudanny
'''


from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_parameters_yield_rawdata.csv", header=0, encoding = 'big5')
# Filter the data by material number and layers
data = data.loc[(data[u'料號'] == 'A58Z185') & (data[u'層別名稱'] == u'外層')]
data = data.reset_index(drop=True)

plate_num = pd.DataFrame(data[u"排板數"].str.split('*',2).tolist(),
                                   columns = ['plate_num1','plate_num2', "plate_num3"])
plate_num = plate_num.astype('int')
plate_ttl = plate_num["plate_num1"] * plate_num["plate_num2"] * plate_num["plate_num3"]
plate_ttl.name = "plate_ttl"
data = data.join(plate_ttl)

#不良PSC總數/(批量*排版數*2)
data["yield"] = data[u"有缺點總數"] / (2.0 * data[u"批量"] * data["plate_ttl"])
np_yield = data["yield"].as_matrix()
yield_avg = np.average(np_yield)
yield_std = np.std(np_yield)

#Draw the yield scatter chart
import matplotlib.pyplot as plt
pd.tools.plotting.scatter_matrix(data.loc[:, "yield":"yield"], diagonal="kde")
plt.tight_layout()
plt.show()

"""
A 不良率<平均值-一倍標準差
B 平均值-一倍標準差≦不良率＜平均值
C 平均值≦不良率＜平均值+一倍標準差
D 平均值+一倍標準差≦不良率
"""
rng1 = yield_avg
rng2 = yield_avg + yield_std * 0.5
rng3 = yield_avg + yield_std * 1
print rng1, rng2, rng3

# define our own classifier
data["yield_classification"] = "D"
data["yield_classification"][(data["yield"] < rng1)] = "A"
data["yield_classification"][(data["yield"] >= rng1) & (data["yield"] < rng2)] = "B"
data["yield_classification"][(data["yield"] >= rng2) & (data["yield"] < rng3)] = "C"


# Group by classification
freq = data.groupby('yield_classification')['yield_classification'].transform('count')

del data[u"顯影"]
del data[u"有缺點總數"]
del data[u"批量"]
del data[u"排板數"]
del data[u"料號"]
del data[u"層別名稱"]
del data[u"批號"]
del data["yield"]
del data["plate_ttl"]
data = data.dropna()

# Encode data columns
from sklearn.preprocessing import LabelEncoder
class_le1 = LabelEncoder()
class_le2 = LabelEncoder()
class_le3 = LabelEncoder()
data[u"壓膜"] = class_le1.fit_transform(data[u"壓膜"])
data[u"前處理"] = class_le2.fit_transform(data[u"前處理"])
data[u"曝光"] = class_le3.fit_transform(data[u"曝光"])


# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Split trining data and testing data
feature_num = data.shape[1]
X = data.iloc[:, 0:feature_num -1]
y = data.iloc[:, feature_num - 1:feature_num].as_matrix().ravel()

# Performing one-hot encoding on nominal features
#X = pd.get_dummies(X).as_matrix()

#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Assessing Feature Importances with Random Forests
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100,
                                random_state=0,
                                n_jobs=-1)
forest.fit(trX, trY)

trY_pred = forest.predict(trX)
teY_pred = forest.predict(teX)
accuracy = forest.score(teX, teY)
print(accuracy)

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=3, 
                                  min_samples_leaf=5, 
                                  random_state=0)
clf = clf.fit(trX, trY)
import pydotplus 
dot_data = tree.export_graphviz(clf,
                                feature_names=["前處理","壓膜","曝光"],
                                class_names=["A","B","C","D"],
                                filled=True,
                                rounded=True,
                                special_characters=True,
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("yield_classification.pdf")


"""
my_max = data['yield'].max()
my_min = data['yield'].min()

my_rng_stage = (my_max - my_min) / 6.0
rng1 = my_min + my_rng_stage * 1.0
rng2 = my_min + my_rng_stage * 2.0
rng3 = my_min + my_rng_stage * 3.0
rng4 = my_min + my_rng_stage * 4.0
rng5 = my_min + my_rng_stage * 5.0

rng1 = 0.01
rng2 = 0.02
rng3 = 0.03
rng4 = 0.04
rng5 = 0.075

# define our own classifier
data["yield_classification"] = "F"
data["yield_classification"][(data["yield"] >= my_min) & (data["yield"] < rng1)] = "A"
data["yield_classification"][(data["yield"] >= rng1) & (data["yield"] < rng2)] = "B"
data["yield_classification"][(data["yield"] >= rng2) & (data["yield"] < rng3)] = "C"
data["yield_classification"][(data["yield"] >= rng3) & (data["yield"] < rng4)] = "D"
data["yield_classification"][(data["yield"] >= rng4) & (data["yield"] < rng5)] = "E"

# Group by classification
data['freq'] = data.groupby('yield_classification')['yield_classification'].transform('count')


def classifier(row):
    if row["yield"] >= my_min and row["yield"] < fst_rng :
        return "A"
    elif row["yield"] >= fst_rng and row["yield"] < snd_rng :
        return "B"
    else:
    return "C"
data["yield_classification"] = data.apply(classifier, axis=1)

# print out the raw data in dataframe
for i in range(data["yield_classification"].shape[0]):
    print data["yield"][i], " , ", data["yield_classification"][i]
"""
"""
# K-means 不良率分群
from sklearn.cluster import KMeans
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(data["yield"])
    distortions.append(km.inertia_)
"""
