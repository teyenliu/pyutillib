'''
Created on Jan 19, 2017

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

# Import MNIST data
data = pd.read_csv("view_parameters_yield_mapping.csv", header=0)

data["C"] = data["C"].fillna("None")
data["D"] = data["D"].fillna("None")
data["E"] = data["E"].fillna("None")
data["F"] = data["F"].fillna("None")
data["I"] = data["I"].fillna("None")
data["J"] = data["J"].fillna("None")
data = data.dropna()

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
data_std = stdsc.fit_transform(data)


if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

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


new_data2 = new_data.as_matrix()
trX = new_data2[:, 0:18]
trY = new_data2[:, 18:19]

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