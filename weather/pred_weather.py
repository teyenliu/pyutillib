# -*- coding: utf-8 -*-
# coding=utf-8

'''
Created on Feb 24, 2017

@author: liudanny
'''
import pandas as pd
import numpy as np
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
# from sklearn import svm
import matplotlib.pyplot as plt
 
# 加载数据集
#url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/GlobalTemperatures.csv'
#climate_data = requests.get(url).content
 
#climate_df = pd.read_csv(io.StringIO(climate_data.decode('utf-8')))  # python2使用StringIO.StringIO
climate_df = pd.read_csv("GlobalTemperatures.csv", header=0, encoding = 'utf-8')  
 
# 把date字段转为日期
climate_df.date = pd.to_datetime(climate_df.date)
# 把日期分成年月日
climate_df['year'] = climate_df['date'].dt.year
climate_df['month'] = climate_df['date'].dt.month
 
climate_df = climate_df.drop('date', 1)
 
# 只使用LandAverageTemperature字段
climate_df = climate_df[np.isfinite(climate_df['LandAverageTemperature'])]  # 地表平均温度
 
climate_df = climate_df.drop('LandMaxTemperature', 1)
climate_df = climate_df.drop('LandMaxTemperatureUncertainty', 1)
climate_df = climate_df.drop('LandMinTemperature', 1)
climate_df = climate_df.drop('LandMinTemperatureUncertainty', 1)
climate_df = climate_df.drop('LandAverageTemperatureUncertainty', 1)
climate_df = climate_df.drop('LandAndOceanAverageTemperature', 1)
climate_df = climate_df.drop('LandAndOceanAverageTemperatureUncertainty', 1)
 
climate_df = climate_df.fillna(-9999)
 
# print(climate_df.head())
# print(climate_df.tail())
 
X = np.array(climate_df.drop(['LandAverageTemperature'], 1))
Y = np.array(climate_df['LandAverageTemperature'])
 
"""
# 测试准确率
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
clf = GradientBoostingRegressor(learning_rate=0.03, max_features=0.03, n_estimators=500)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)  # 0.965156260089
"""
 
#clf = GradientBoostingRegressor(learning_rate=0.03, max_features=0.03, n_estimators=500)
clf = LinearRegression()
clf.fit(X, Y)
 
predict_X = []
start_year = 2016
for year in range(3):
    for month in range(12):
        predict_X.append([start_year + year, month + 1])
 
# 预测2016-2025年温度
pridict_Y = clf.predict(predict_X)
print pridict_Y
 
# 绘制1980-2015年平均温度
year_x = []
Y = np.hstack((Y, pridict_Y))
for x in np.vstack((X, predict_X)):
    year_x.append(x[0])
data = {}
for x, y in zip(year_x, Y):
    if x not in data.keys():
        data[x] = y
    else:
        data[x] = (data[x] + y) 
for key, value in data.items():
    if key > 1980:
        plt.scatter(key, value / 12)
 
plt.show()
