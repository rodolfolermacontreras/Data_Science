# SpuriousCorrelation.py
# Copyright 2018 by Ernst Henle

# Spurious self-correlation
# Virtual Correlation
# https://en.wikipedia.org/wiki/Spurious_correlation
# Pearson, Karl (1897). "Mathematical Contributions to the Theory of 
# Evolution-On a Form of Spurious Correlation Which May Arise When Indices Are
# Used in the Measurement of Organs". Proceedings of the Royal Society of 
# London. 60: 489-498
# Reed J. L. (1921) "On the correlation between any two functions and its 
# application to the genaral case of spurious correlation," J. of the 
# Washington Academy of Science, Vol 11 pp 449-455

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create two random variables that should have no correlation with each other
x = 10*np.random.random(1000) + 10
y = 10*np.random.random(1000) + 10
Correlation = np.corrcoef(x, y)[0,1].round(3)

plt.figure(1, figsize=(5,5), facecolor='lightgrey')
plt.plot(x, y, 'bo', ms=4, mfc='none')
font = {'weight': 'bold', 'size': 16,}
Label = "No correlation between x and y\n (r = " + str(Correlation) + ")"
plt.title(Label, fontdict=font)

# Create a third random variable that has no correlation with either of the
# first two variables
z = 10*np.random.random(1000) + 10
Correlation = np.corrcoef(x, z)[0,1].round(3)
print("Correlation between x and z:", Correlation)
Correlation = np.corrcoef(y, z)[0,1].round(3)
print("Correlation between y and z:", Correlation)

# Create ratios of x per z and y per z
xz = x/z
yz = y/z
Correlation = np.corrcoef(xz, yz)[0,1].round(3)
regr = LinearRegression()
regr.fit(np.array(xz, ndmin=2).T, yz)

# Plot Correlated ratios
plt.figure(2, figsize=(5,5), facecolor='lightgrey')
plt.plot(xz, yz, 'bo', ms=4, mfc='none')
font = {'weight': 'bold', 'size': 16,}
Label = "x/z and y/z are correlated\n (r = " + str(Correlation) + ")"
plt.title(Label, fontdict=font)
xline = np.array([0.5, 2.0])
yline = regr.intercept_ + regr.coef_[0] * xline
line = plt.plot(xline, yline)
plt.setp(line, 'color', 'k', 'linewidth', 4.0, 'ls', '--')
