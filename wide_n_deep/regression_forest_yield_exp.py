
# -*- coding: big5 -*-
# coding=Big5
from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_parameters_yield_mapping3_new.csv", header=0, encoding = 'big5')
#data.columns = [u"§@·~½u§O",u"³Ì¤p½u®|",u"³Ì¤p½u¶Z",u"À£½¤³t«×",
#                u"¹qÁá®É¶¡",u"¹q¬y(A)",u"Avg",u"¼Ð·Ç®t",u"¤£¨}²v"]
data = data.dropna()

# Split trining data and testing data
feature_num = data.shape[1]
X = data.iloc[:, 0:feature_num -1]
y = data.iloc[:, feature_num - 1:feature_num].as_matrix().ravel()

# Performing one-hot encoding on nominal features
X = pd.get_dummies(X).as_matrix()


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
from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X, y, test_size=0.3, random_state=0)


# Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
slr = LinearRegression()
#slr = GradientBoostingRegressor(learning_rate=0.03, max_features=0.03, n_estimators=500)
from sklearn.model_selection import train_test_split


slr.fit(trX, trY)
trY_pred = slr.predict(trX)
teY_pred = slr.predict(teX)
accuracy = slr.score(teX, teY)
print(accuracy)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(trY, trY_pred),
        mean_squared_error(teY, teY_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(trY, trY_pred),
        r2_score(teY, teY_pred)))


#======================== polynomial regression ============================ 
# -*- coding: big5 -*-
# coding=Big5
from sklearn import preprocessing
import numpy as np
import pandas as pd
# Import data
data = pd.read_csv("view_parameters_yield_mapping3_new.csv", header=0, encoding = 'big5')
#data.columns = [u"§@·~½u§O",u"³Ì¤p½u®|",u"³Ì¤p½u¶Z",u"À£½¤³t«×",
#                u"¹qÁá®É¶¡",u"¹q¬y(A)",u"Avg",u"¼Ð·Ç®t",u"¤£¨}²v"]
data = data.dropna()

# Split trining data and testing data
feature_num = data.shape[1]
X = data.iloc[:, 0:feature_num -1]
y = data.iloc[:, feature_num - 1:feature_num].as_matrix().ravel()


# Performing one-hot encoding on nominal features
X = pd.get_dummies(X).as_matrix()

from sklearn.preprocessing import PolynomialFeatures
# create quadratic and features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(
    X_poly, y, test_size=0.3, random_state=0)

# Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
regr = slr.fit(trX, trY)

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


#============================ RandomForestRegressor ==========================
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=100, 
                               criterion='mse',
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

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
# ====================================================================
import matplotlib.pyplot as plt
plt.figure()
#plt.subplot(211)
plt.plot(teY[:], 'bo', teY_pred[:], 'k')
plt.legend(['Unitech testing', 'Predicted prediction'], loc='upper left')
plt.show()


#import csv
#writer = csv.writer(open("results.csv", 'w'))
#for i in range(teY.shape[0]):
#    writer.writerow(teY[i])

np.savetxt('results.csv', (np.reshape(teY,(-1,1)), np.reshape(teY_pred,(-1,1))), delimiter=',')
        

        