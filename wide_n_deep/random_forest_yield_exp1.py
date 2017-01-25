

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv("view_yields2.csv", header=0)
data = data.dropna()


data.columns = ["E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W"]
feat_labels = data.columns[0:-1]


# Encode data columns
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
data['F'] = class_le.fit_transform(data['F'])
data['G'] = class_le.fit_transform(data['G'])
data['H'] = class_le.fit_transform(data['H'])

# Split trining data and testing data
X = data.iloc[:, 0:18].as_matrix()
y = data.iloc[:, 18:19].as_matrix()

from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.3, random_state=0)


# Assessing Feature Importances with Random Forests
"""
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(trX, trY)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(trX.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(trX.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(trX.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, trX.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()
"""

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf = clf.fit(trX, trY)

with open("yields.dot", 'w') as f:
    f = tree.export_graphviz(clf,
                             out_file=f)

import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("yields.pdf") 
