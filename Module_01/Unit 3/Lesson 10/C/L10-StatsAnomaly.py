"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from scipy.stats import norm
#################

# Auxiliary function
def z_score(x):
	m = np.mean(x)
	s = np.std(x)
	return (np.array(x) - m) / s
##################

# Initialization
rnd.seed(a = 123)
m1, s1, n1 = 0.0, 1.0, 30
m2, s2, n2 = 7.0, 0.5, 20
nb = 30 # number of bins for histogram
n = n1 + n2 # total number of data points in feature
th1 = 0.05 # p-value threshold below which a data point is considered an outlier for 1-D data
th2 = 0.01 # p-value threshold below which a data point is considered an outlier for 2-D data
##################

# Data to work with (X = values of feature X, Z = 2-dimensional feature set)
X = np.hstack([np.random.normal(m1, s1, n1), np.random.normal(m2, s2, n2)])
X[0] = 3.9 # inlier
X[-1] = 10.0 # outlier 1
X[1] = -5.0 # outlier 2
Z = np.transpose(np.vstack([np.random.normal(m1, s1, n), np.random.normal(m2, s2, n)]))
Z[0,0] = 3.5 # outlier 1
Z[0,1] = 5.0 # outlier 1
Z[-1,0] = -3.0 # outlier 2
Z[-1,1] = 9.0 # outlier 2
nf = 2 # number of features in Z
##################

# Explore anomalies in 1-D space
# Take a look at the histogram of this data
plt.figure()
plt.title('Histogram of Feature X')
plt.xlim([np.min(X)-0.2, np.max(X)+0.2])
plt.xlabel('Value ranges of feature X')
plt.ylabel('Relative frequency')
hist, bins = np.histogram(X, bins=nb)
w = 0.7 * (bins[1] - bins[0]) # width of each bin
c = (bins[:-1] + bins[1:]) / 2 # center point of histogram
plt.bar(c, hist, align='center', width=w)
plt.show()
##################

# Identify anomalies in feature x
zx = z_score(X) # z-scores for various points in X
px = norm.sf(abs(zx)) # p-value based on z-score (one-sided approach)
outliers = X[px < th1]
print ('\nOutliers for feature X:', outliers, '\n')
#################

# Explore anomalies in 2-D space
# Take a look at the scatter plot of this data
plt.figure()
plt.title('Scatter Plot of Featureset Z')
plt.xlabel('Values of 1st feature')
plt.ylabel('Values of 2nd feature')
plt.scatter(Z[:,0], Z[:,1])
plt.show()
################

# Identify anomalies in featureset Z using the Multivariate Gaussian Distribution approach
S = np.cov(Z[:,0], Z[:,1]) # covariance matrix for the features in Z
S_inv = np.linalg.inv(S) # inverse of covariance matrix (useful for assessing the multivariate p-value)
m = np.mean(Z, 0) # mean feature values 
d = Z - m # semi-normalized feature values, using the means
c = ((2*np.pi)**(nf/2) * np.sqrt(np.linalg.det(S)))**(-1) # coefficient for p-value calculations
p = np.zeros(n)

for i in range(n):
	p[i] = c * np.exp(-0.5 * np.dot(np.dot(d[i], S_inv), np.transpose(d[i])))

outliers = Z[p < th2]
print ('Outliers for featureset Z:\n',outliers, '\n')
#################
