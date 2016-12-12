'''
Created on Dec 9, 2016

@author: liudanny
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
np.random.seed(101)
y = np.random.rand(10)

small = [i for i in range(len(x)) if y[i] < .5 ]
big = [i for i in range(len(x)) if y[i] > .5 ]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x,y)
# user wants to plot lines connecting big values (>.5)
ax.plot(x[big],y[big])

plt.show()
# now user wants to delete the first (and only) line
del ax.lines[0]

# so that they can plot a line only showing small values
ax.plot(x[small],y[small])

plt.show()