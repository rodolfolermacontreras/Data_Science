# SimpsonsParadox.py
# Copyright 2018 by Ernst Henle

# Simpsons Paradox
# A consistent trend appears in multiple datasets
# The trend reverses when these datasets are combined
# https://en.wikipedia.org/wiki/Simpson's_paradox

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data set 0
x0 = np.random.normal(loc=10, scale=4, size=100)
y0 = 80 + x0 + np.random.normal(loc=10, scale=4, size=100)
Corr0 = 'r = ' + str(np.corrcoef(x0, y0)[0,1].round(3))
regr = LinearRegression()
regr.fit(np.array(x0, ndmin=2).T, y0)
xline = np.array([0, 50])
yline0 = regr.intercept_ + regr.coef_[0] * xline

# Data set 1
x1 = np.random.normal(loc=20, scale=4, size=100)
y1 = 40 + x1 + np.random.normal(loc=10, scale=4, size=100)
Corr1 = 'r = ' + str(np.corrcoef(x1, y1)[0,1].round(3))
regr = LinearRegression()
regr.fit(np.array(x1, ndmin=2).T, y1)
xline = np.array([0, 50])
yline1 = regr.intercept_ + regr.coef_[0] * xline

# Data set 2
x2 = np.random.normal(loc=30, scale=4, size=100)
y2 = 0 + x2 + np.random.normal(loc=10, scale=4, size=100)
Corr2 = 'r = ' + str(np.corrcoef(x2, y2)[0,1].round(3))
regr = LinearRegression()
regr.fit(np.array(x2, ndmin=2).T, y2)
xline = np.array([0, 50])
yline2 = regr.intercept_ + regr.coef_[0] * xline

# Combined Data set
x = np.concatenate((x0, x1, x2))
y = np.concatenate((y0, y1, y2))
Corr = 'r = ' + str(np.corrcoef(x, y)[0,1].round(3))
regr = LinearRegression()
regr.fit(np.array(x, ndmin=2).T, y)
xline = np.array([0, 50])
yline = regr.intercept_ + regr.coef_[0] * xline

plt.figure(1, figsize=(5,5), facecolor='lightgrey')
plt.xlim(0,50)
plt.ylim(0,120)
plt.plot(x0, y0, 'ro', ms=8, mfc='none')
plt.plot(x1, y1, 'y^', ms=8, mfc='none')
plt.plot(x2, y2, 'bs', ms=8, mfc='none')
line = plt.plot(xline, yline)
plt.setp(line, 'color', 'k', 'linewidth', 2.0, 'ls', '--')
plt.legend([Corr0, Corr1, Corr2, Corr], facecolor='lightgrey')
line0 = plt.plot(xline, yline0)
plt.setp(line0, 'color', 'r', 'linewidth', 1.0)
line1 = plt.plot(xline, yline1)
plt.setp(line1, 'color', 'y', 'linewidth', 1.0)
line2 = plt.plot(xline, yline2)
plt.setp(line2, 'color', 'b', 'linewidth', 1.0)
