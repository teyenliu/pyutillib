'''
Created on Mar 30, 2017

@author: liudanny
'''
# Feature Importance
from sklearn import datasets
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()



# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import pandas as pd

# Import data
data = pd.read_csv("view_parameters_yield_mapping.csv", header=0)

data["C"] = data["C"].fillna("None")
data["D"] = data["D"].fillna("None")
data["E"] = data["E"].fillna("None")
data["F"] = data["F"].fillna("None")
data["I"] = data["I"].fillna("None")
data["J"] = data["J"].fillna("None")
data = data.dropna()

# Encode data columns
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
data['A'] = class_le.fit_transform(data['A'])
data['B'] = class_le.fit_transform(data['B'])
data['C'] = class_le.fit_transform(data['C'])
data['D'] = class_le.fit_transform(data['D'])
data['E'] = class_le.fit_transform(data['E'])
data['F'] = class_le.fit_transform(data['F'])
data['I'] = class_le.fit_transform(data['I'])
data['J'] = class_le.fit_transform(data['J'])

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
data_std = stdsc.fit_transform(data)

#if Version(sklearn_version) < '0.18':
    from sklearn.model_selection import train_test_split
#else:
#    from sklearn.model_selection import train_test_split

X, y = data_std.iloc[:, 0:19].values, data_std.iloc[:, 19:20].values
trX, teX, trY, teY = \
    train_test_split(X, y, test_size=0.2, random_state=0)

train, test = \
    train_test_split(data, test_size=0.2, random_state=0)

# Write our changed dataframes to csv.
test.to_csv("./test.csv", index=False)
train.to_csv('./train.csv', index=False)


"""
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(data)
imputed_data = imr.transform(data.values)
imputed_data
"""

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
data_std = stdsc.fit_transform(data)
trX = data_std[:, 0:19]
trY = data_std[:, 19:20]



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

clf = MultiTaskElasticNetCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.23)
sfm.fit(trX, trY)
new_trX = sfm.transform(trX)
n_features = new_trX.shape[1]
print sfm.get_support(indices=True)




pd.tools.plotting.scatter_matrix(ttl_X, diagonal="kde")
plt.tight_layout()
plt.show()

ax = ttl_X[["K","L","M","O","P","Q","R","S"]].plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
plt.show()


import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML

corrmat = ttl_X.corr()
sns.heatmap(corrmat, vmax=1., square=False).xaxis.tick_top()
plt.show()

sns.lmplot("R", "S", ttl_X, hue="Q", fit_reg=False)
plt.show()

================================================================================================

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Building a decision tree

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(trX, trY)


import pydotplus
from IPython.display import Image
from IPython.display import display

import pydotplus
from sklearn.tree import export_graphviz

dot_data = export_graphviz(
tree, 
out_file=None,
# the parameters below are new in sklearn 0.18
feature_names=['petal length', 'petal width'],  
class_names=['setosa', 'versicolor', 'virginica'],  
filled=True,
rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)  
display(Image(graph.create_png()))

export_graphviz(tree, 
                out_file='tree.dot', 
                feature_names=['petal length', 'petal width'])
Image(filename='./images/03_18.png', width=600)



from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf = clf.fit(iris.data, iris.target)

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 

