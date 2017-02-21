
# -*- coding: big5 -*-
# coding=Big5
from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_parameters_yield_mapping3_new.csv", header=0, encoding = 'big5')
data.columns = [u"作業線別",u"最小線徑",u"最小線距",u"壓膜速度",
                u"電鍍時間",u"電流(A)",u"Avg",u"標準差",u"不良率"]
data.dropna()
feat_labels = data.columns[0:-1]


# Visualizing the important characteristics of a dataset
import matplotlib.pyplot as plt
cols = data.columns[1:-1]
pd.tools.plotting.scatter_matrix(data[cols], diagonal="kde")
plt.tight_layout()
plt.show()

ax = data[cols].plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
plt.show()

import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML

corrmat = data[cols].corr()
sns.heatmap(corrmat, vmax=1., cbar=True,annot=True,square=True).xaxis.tick_top()
plt.show()

# Split trining data and testing data
#X = preprocessing.scale(data.iloc[:, 0:7].as_matrix())
data = data.dropna()
X = data.iloc[:, 1:8].as_matrix()
y = data.iloc[:, 8:9].as_matrix().ravel()

from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.2, random_state=0)


# Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
from sklearn.model_selection import train_test_split


slr.fit(trX, trY)
trY_pred = slr.predict(trX)
teY_pred = slr.predict(teX)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(trY, trY_pred),
        mean_squared_error(teY, teY_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(trY, trY_pred),
        r2_score(teY, teY_pred)))


#======================== polynomial regression ============================ 
from sklearn.preprocessing import PolynomialFeatures
# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
trX_quad = quadratic.fit_transform(trX)
trX_cubic = cubic.fit_transform(trX)


regr = slr.fit(trX_quad, trY)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))


#============================ RandomForestRegressor ==========================
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=100, 
                               criterion='mse',
                               alpha=0.1, 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(trX, trY)

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(trX, trY)

trY_pred = forest.predict(trX)
teY_pred = forest.predict(teX)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(trY, trY_pred),
        mean_squared_error(teY, teY_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(trY, trY_pred),
        r2_score(teY, teY_pred)))

#============================ Lasso ==========================
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

lasso.fit(trX, trY)
trY_pred = lasso.predict(trX)
teY_pred = lasso.predict(teX)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(trY, trY_pred),
        mean_squared_error(teY, teY_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(trY, trY_pred),
        r2_score(teY, teY_pred)))

#============================ ElasticNet ==========================
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
enet.fit(trX, trY)
trY_pred = enet.predict(trX)
teY_pred = enet.predict(teX)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(trY, trY_pred),
        mean_squared_error(teY, teY_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(trY, trY_pred),
        r2_score(teY, teY_pred)))
