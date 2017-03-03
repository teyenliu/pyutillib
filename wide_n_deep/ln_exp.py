# -*- coding: utf-8 -*-
# coding=utf-8
import numpy as np

# 設定樣本資料數為300筆
number_of_points = 300
x_point = []
y_point = []
w = 0.25
b = 0.33

# 產生樣本資料(x,y)
for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5)
    # 增加雜訊來使樣本資料更為逼真
    y = w * x + b + np.random.normal(0.0, 0.1)
    x_point.append([x])
    y_point.append([y])

# 繪出點圖
import matplotlib.pyplot as plt

plt.plot(x_point, y_point, 'og', label='linear regression example')
plt.legend()
plt.show()

import tensorflow as tf
# 定義TensorFlow變數
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))

# 定義資料模型
y = W * x_point + B

# 成本函數
cost_function = tf.reduce_mean(tf.square(y - y_point))
# 使用梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(cost_function)

# 初始化所有變數
model = tf.global_variables_initializer()

# 建立TensorFlow Session
with tf.Session() as session:
        session.run(model)
        # 迭代25次,每5次就印出點圖與計算出的線性回歸的線
        for step in range(0, 26):
                session.run(train)
                if (step % 5) == 0:
                        plt.plot(x_point, y_point,
                                 'og', label='step = {}'.format(step))
                        plt.plot(x_point,
                                 session.run(W) * x_point + session.run(B))
                        plt.legend()
                        plt.show()
