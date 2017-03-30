'''
Created on Mar 30, 2017

@author: liudanny
'''

============================================ Part I ==============================================
# -*- coding: big5 -*-
# coding=Big5
from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_yields2.csv", header=0, encoding = 'big5')
data.columns = [u"層別名稱",u"結構分類",u"EL線別",u"PL產出線別",u"FP產出線別",
                u"壓膜機",u"曝光機",u"顯影線",u"蝕刻線",u"不良率分類"]
data.dropna()
feat_labels = data.columns[0:-1]

# Encode data columns
from sklearn.preprocessing import LabelEncoder
class_le1 = LabelEncoder()
class_le2 = LabelEncoder()
class_le3 = LabelEncoder()
class_le4 = LabelEncoder()
class_le5 = LabelEncoder()
class_le6 = LabelEncoder()
class_le7 = LabelEncoder()
class_le8 = LabelEncoder()
class_le9 = LabelEncoder()
data[u"層別名稱"] = class_le1.fit_transform(data[u"層別名稱"])
data[u"結構分類"] = class_le2.fit_transform(data[u"結構分類"])
data[u"EL線別"] = class_le3.fit_transform(data[u"EL線別"])
data[u"PL產出線別"] = class_le4.fit_transform(data[u"PL產出線別"])
data[u"FP產出線別"] = class_le5.fit_transform(data[u"FP產出線別"])
data[u"壓膜機"] = class_le6.fit_transform(data[u"壓膜機"])
data[u"曝光機"] = class_le7.fit_transform(data[u"曝光機"])
data[u"顯影線"] = class_le8.fit_transform(data[u"顯影線"])
data[u"蝕刻線"] = class_le9.fit_transform(data[u"蝕刻線"])

# Split trining data and testing data
#X = preprocessing.scale(data.iloc[:, 0:7].as_matrix())
X = data.iloc[:, 0:9].as_matrix()
y = data.iloc[:, 9:10].as_matrix().ravel()

from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import tree

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=100,
                                random_state=1,
                                n_jobs=-1)


pipe_lr = Pipeline([('clf', forest)])

pipe_lr.fit(trX, trY)
print('Test Accuracy: %.3f' % pipe_lr.score(teX, teY))

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(trX.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

# Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()
clf = clf.fit(trX, trY)
importances = clf.feature_importances_ 
indices = np.argsort(importances)[::-1]

for f in range(trX.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))


model = SelectFromModel(clf, prefit=True)
trX_new = model.transform(trX)
trX_new.shape               
print model.get_support(indices=True)

# Re-print the test accuracy
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=100,
                                random_state=1,
                                n_jobs=-1)


pipe_lr = Pipeline([('clf', forest)])

pipe_lr.fit(trX_new, trY)
print('Test Accuracy: %.3f' % pipe_lr.score(trX_new, teY))

                    
# Print Decision Tree                
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=4, 
                                  min_samples_leaf=5, 
                                  random_state=0)
clf = clf.fit(trX, trY)

import pydotplus 
dot_data = tree.export_graphviz(clf,
                                feature_names=["層別名稱","結構分類","EL線別",
                                               "PL產出線別","FP產出線別",
                                               "壓膜機","曝光機","顯影線","蝕刻線"],
                                class_names=["A","B","C","D","E","F"],
                                filled=True,
                                rounded=True,
                                special_characters=True,
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("yields.pdf")
======================================== Part II =====================================================
# -*- coding: big5 -*-
# coding=Big5
from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_parameters_yield_mapping2.csv", header=0, encoding = 'big5')
data.columns = [u"層別名稱",u"前處理",u"壓膜",u"曝光(上機)",u"顯影",u"排板數",
                u"產品型態",u"最小線徑",u"最小線距",u"出貨面積",u"乾膜種類",
                u"壓膜方式",u"壓膜速度",u"壓膜機台",u"曝光機台",u"E1廠別",
                u"E1線別",u"格數",u"乾膜尺寸",u"銅厚均值",u"銅厚數據",
                u"磨刷段傳送速度",u"曝光D值",u"顯影速度",u"不良率分類"]

data[u"最小線徑"].fillna(data[u"最小線徑"].mean(), inplace=True)
data[u"最小線距"].fillna(data[u"最小線距"].mean(), inplace=True)
data[u"出貨面積"].fillna(data[u"出貨面積"].mean(), inplace=True)
data[u"壓膜速度"].fillna(data[u"壓膜速度"].mean(), inplace=True)
data[u"乾膜尺寸"].fillna(data[u"乾膜尺寸"].mean(), inplace=True)
data[u"銅厚均值"].fillna(data[u"銅厚均值"].mean(), inplace=True)
data[u"銅厚數據"].fillna(data[u"銅厚數據"].mean(), inplace=True)
data[u"磨刷段傳送速度"].fillna(data[u"磨刷段傳送速度"].mean(), inplace=True)
data[u"曝光D值"].fillna(data[u"曝光D值"].mean(), inplace=True)
data[u"顯影速度"].fillna(data[u"顯影速度"].mean(), inplace=True)
data.fillna("NAN")
feat_labels = data.columns[0:-1]

# Encode data columns
from sklearn.preprocessing import LabelEncoder
class_le1 = LabelEncoder()
class_le2 = LabelEncoder()
class_le3 = LabelEncoder()
class_le4 = LabelEncoder()
class_le5 = LabelEncoder()
class_le6 = LabelEncoder()
class_le7 = LabelEncoder()
class_le8 = LabelEncoder()
class_le9 = LabelEncoder()
class_le10 = LabelEncoder()
class_le11 = LabelEncoder()
class_le12 = LabelEncoder()
class_le13 = LabelEncoder()
class_le14 = LabelEncoder()
data[u"層別名稱"] = class_le1.fit_transform(data[u"層別名稱"])
data[u"前處理"] = class_le2.fit_transform(data[u"前處理"])
data[u"壓膜"] = class_le3.fit_transform(data[u"壓膜"])
data[u"曝光(上機)"] = class_le4.fit_transform(data[u"曝光(上機)"])
data[u"顯影"] = class_le5.fit_transform(data[u"顯影"])
data[u"排板數"] = class_le6.fit_transform(data[u"排板數"])
data[u"產品型態"] = class_le7.fit_transform(data[u"產品型態"])
data[u"乾膜種類"] = class_le8.fit_transform(data[u"乾膜種類"])
data[u"壓膜方式"] = class_le9.fit_transform(data[u"壓膜方式"])
data[u"壓膜機台"] = class_le10.fit_transform(data[u"壓膜機台"])
data[u"曝光機台"] = class_le11.fit_transform(data[u"曝光機台"])
data[u"E1廠別"] = class_le12.fit_transform(data[u"E1廠別"])
data[u"E1線別"] = class_le13.fit_transform(data[u"E1線別"])
data[u"格數"] = class_le14.fit_transform(data[u"格數"])


# Split trining data and testing data
#X = preprocessing.scale(data.iloc[:, 0:20].as_matrix())
X = data.iloc[:, 0:24].as_matrix()
y = data.iloc[:, 24:25].as_matrix().ravel()

from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.2, random_state=0)

    
# Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()
clf = clf.fit(trX, trY)
importances = clf.feature_importances_ 
indices = np.argsort(importances)[::-1]

for f in range(trX.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

                            
model = SelectFromModel(clf, prefit=True)
trX_new = model.transform(trX)
trX_new.shape               
print model.get_support(indices=True)

    
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import tree

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=1000,
                                random_state=1,
                                n_jobs=-1)


pipe_lr = Pipeline([#('scl', StandardScaler()),
                    #('pca', PCA(n_components=4)),
                    ('clf', forest)])

pipe_lr.fit(trX, trY)
print('Test Accuracy: %.3f' % pipe_lr.score(teX, teY))


from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=4, 
                                  min_samples_leaf=5, 
                                  random_state=0)

clf = clf.fit(trX, trY)

import pydotplus 
dot_data = tree.export_graphviz(clf,
                                feature_names=["層別名稱","前處理","壓膜","曝光(上機)",
                                                "顯影","排板數","產品型態","最小線徑",
                                                "最小線距","出貨面積","乾膜種類","壓膜方式",
                                                "壓膜速度","壓膜機台","曝光機台","E1廠別",
                                                "E1線別","格數","乾膜尺寸","銅厚均值","銅厚數據",
                                                "磨刷段傳送速度","曝光D值","顯影速度"],
                                class_names=["A","B","C","D","E","F"],
                                filled=True,
                                rounded=True,
                                special_characters=True,
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("yields.pdf")


# Split trining data and testing data
from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Assessing Feature Importances with Random Forests
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000,
                                random_state=1,
                                n_jobs=-1)
forest.fit(trX, trY)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(trX.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
                            
======================================== Part III =====================================================
# -*- coding: big5 -*-
# coding=Big5
from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_parameters_yield_mapping3.csv", header=0, encoding = 'big5')
data.columns = [u"層別名稱",u"排板數",u"最小線徑",u"最小線距",u"出貨面積",
                u"乾膜種類",u"壓膜方式",u"壓膜速度",u"壓膜機台",u"曝光機台",
                u"E1廠別",u"E1線別",u"銅厚規格",u"尺寸",u"板厚(mil)",
                u"電鍍時間",u"電流(A)",u"Avg",u"Max",u"Min",u"R值",
                u"標準差",u"不良率分類"]

data[u"最小線徑"].fillna(data[u"最小線徑"].mean(), inplace=True)
data[u"最小線距"].fillna(data[u"最小線距"].mean(), inplace=True)
data[u"出貨面積"].fillna(data[u"出貨面積"].mean(), inplace=True)
data[u"壓膜速度"].fillna(data[u"壓膜速度"].mean(), inplace=True)
data[u"電鍍時間"].fillna(data[u"電鍍時間"].mean(), inplace=True)
data[u"電流(A)"].fillna(data[u"電流(A)"].mean(), inplace=True)
data[u"Avg"].fillna(data[u"Avg"].mean(), inplace=True)
data[u"Max"].fillna(data[u"Max"].mean(), inplace=True)
data[u"Min"].fillna(data[u"Min"].mean(), inplace=True)
data[u"R值"].fillna(data[u"R值"].mean(), inplace=True)
data[u"標準差"].fillna(data[u"標準差"].mean(), inplace=True)

data.fillna("NAN")
feat_labels = data.columns[0:-1]

# Encode data columns
from sklearn.preprocessing import LabelEncoder
class_le1 = LabelEncoder()
class_le2 = LabelEncoder()
class_le3 = LabelEncoder()
class_le4 = LabelEncoder()
class_le5 = LabelEncoder()
class_le6 = LabelEncoder()
class_le7 = LabelEncoder()
class_le8 = LabelEncoder()
class_le9 = LabelEncoder()
class_le10 = LabelEncoder()
data[u"層別名稱"] = class_le1.fit_transform(data[u"層別名稱"])
data[u"排板數"] = class_le2.fit_transform(data[u"排板數"])
data[u"乾膜種類"] = class_le3.fit_transform(data[u"乾膜種類"])
data[u"壓膜方式"] = class_le4.fit_transform(data[u"壓膜方式"])
data[u"壓膜機台"] = class_le5.fit_transform(data[u"壓膜機台"])
data[u"曝光機台"] = class_le6.fit_transform(data[u"曝光機台"])
data[u"E1廠別"] = class_le7.fit_transform(data[u"E1廠別"])
data[u"E1線別"] = class_le8.fit_transform(data[u"E1線別"])
data[u"銅厚規格"] = class_le9.fit_transform(data[u"銅厚規格"])
data[u"尺寸"] = class_le10.fit_transform(data[u"尺寸"])
data[u"板厚(mil)"] = class_le10.fit_transform(data[u"板厚(mil)"])

# Split trining data and testing data
#X = preprocessing.scale(data.iloc[:, 0:20].as_matrix())
X = preprocessing.scale(data.iloc[:, 0:22].as_matrix())
y = data.iloc[:, 22:23].as_matrix().ravel()

from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import tree


forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=100,
                                random_state=1,
                                n_jobs=-1)

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', forest)])


pipe_lr.fit(trX, trY)
print('Test Accuracy: %.3f' % pipe_lr.score(teX, teY))