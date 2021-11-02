# L07-2-KMeansNorm_Incomplete.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create Points to cluster
Points = pd.DataFrame()
Points.loc[:,0] = [243,179,152,255,166,162,233,227,204,341,283,202,217,197,191,114,153,215,
      196,187,127,85,182,172,184,252,193,191,187,193,197,200,186,188,155,-99,
      22,68,167,-75,30,49,63,45,58,52,164,51,49,68,52,43,68,72,-51,59,56,-127,
      33,68,143,-26,-85,84,11,105,62,47,-75,2,67,-41,-33,10,28,23,34,19,13,6,
      -73,155,30]
Points.loc[:,1] = [2.1,4,2.6,2.1,2.5,0.4,0.3,4.9,1.1,1,-1.5,3.3,2.2,1.9,2.4,2.2,0.9,1.8,1.7,
      3.2,2.4,4.4,1.4,4.4,2.6,0.6,2.9,3.8,2.6,8.5,8.8,7.5,8.3,8.5,3.5,6.3,-1.4,
      -0.4,3,-5.2,-2.7,-3.2,-0.8,-3.9,-0.6,0.9,-5.1,-2.2,-0.3,-1.2,0.1,-2.1,
      -2.1,3.7,11.8,0,0,-6.6,-1,10.1,11.9,-3,-22,-18.2,-13.3,-8.4,-21.7,-16.7,
      -13.8,-13.9,-13.2,-14.9,-21.6,-16.4,-14.4,-15.8,-15.3,-15.3,-2.7,-13.2,
      -8.9,-3.3,-12.9]

# Create initial cluster centroids
ClusterCentroidGuesses = pd.DataFrame()
ClusterCentroidGuesses.loc[:,0] = [100, 200, 0]
ClusterCentroidGuesses.loc[:,1] = [2, -2, 0]

def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0], ClusterCentroids.loc[LabelNumber,1], s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()

def KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2):
    PointsNorm = Points.copy()
    ClusterCentroids = ClusterCentroidGuesses.copy()
    if NormD1:
        # Determine mean of 1st dimension
        print(" Replace this line with code")
        # Determine standard deviation of 1st dimension
        print(" Replace this line with code")
        # Normalize 1st dimension of Points
        print(" Replace this line with code")
        # Normalize 1st dimension of ClusterCentroids
        print(" Replace this line with code")
    if NormD2:
        # Determine mean of 2nd dimension
        print(" Replace this line with code")
        # Determine standard deviation of 2nd dimension
        print(" Replace this line with code")
        # Normalize 2nd dimension of Points
        print(" Replace this line with code")
        # Normalize 2nd dimension of ClusterCentroids
        print(" Replace this line with code")
    # Do actual clustering
    kmeans = KMeans(n_clusters=3, init=ClusterCentroidGuesses, n_init=1).fit(PointsNorm)
    Labels = kmeans.labels_
    ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)
    if NormD1:
        # Denormalize 1st dimension
        print(" Replace this line with code")
    if NormD2:
        # Denormalize 2nd dimension
        print(" Replace this line with code")
    return Labels, ClusterCentroids

# Compare distributions of the two dimensions
plt.rcParams["figure.figsize"] = [6.0, 4.0] # Standard
plt.hist(Points.loc[:,0], bins = 20, color=[0, 0, 1, 0.5])
plt.hist(Points.loc[:,1], bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare Distributions")
plt.show()

# Change the plot dimensions
plt.rcParams["figure.figsize"] = [8, 8] # Square
# plt.rcParams["figure.figsize"] = [8, 0.5] # Wide
# plt.rcParams["figure.figsize"] = [0.5, 8] # Tall

# Cluster without normalization
# Are the points separated into clusters along one or both dimensions?
# Which dimension separates the points into clusters?
# Set Normalizations
NormD1=False
NormD2=False
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2)
Title = 'No Normalization'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

# Cluster with 1st dimension normalized
# Set Normnalizations
print(" Replace this line with code")
print(" Replace this line with code")
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2)
Title = 'Normalization in first dimension'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

# Cluster with 2nd dimension normalized
# Set Normnalizations
print(" Replace this line with code")
print(" Replace this line with code")
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2)
Title = 'Normalization in second dimension'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

# Cluster with both dimensions normalized
# Set Normnalizations
print(" Replace this line with code")
print(" Replace this line with code")
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2)
Title = 'Normalization in both dimensions'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)
