# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:15:04 2020

@author: Rodolfo Lerma

KMeans Script

General information about the "imports-85":
    After cleaning and normalizing data: 44 (originally we had 26 attributes/columns)
    Original Number of observations: 205
    The data consists of 3 types of entities:
        1) Specifications of a car 
        2) Insurance risk rating
        3) Normalized losses in use as compared to other cars

     Attribute:                Attribute Range:
     ------------------        -----------------------------------------------
  1. symboling:                -3, -2, -1, 0, 1, 2, 3.
  2. normalized-losses:        continuous from 65 to 256.
  3. make:                     alfa-romero, audi, bmw, chevrolet, dodge, honda,
                               isuzu, jaguar, mazda, mercedes-benz, mercury,
                               mitsubishi, nissan, peugot, plymouth, porsche,
                               renault, saab, subaru, toyota, volkswagen, volvo
  4. fuel-type:                diesel, gas.
  5. aspiration:               std, turbo.
  6. num-of-doors:             four, two.
  7. body-style:               hardtop, wagon, sedan, hatchback, convertible.
  8. drive-wheels:             4wd, fwd, rwd.
  9. engine-location:          front, rear.
 10. wheel-base:               continuous from 86.6 120.9.
 11. length:                   continuous from 141.1 to 208.1.
 12. width:                    continuous from 60.3 to 72.3.
 13. height:                   continuous from 47.8 to 59.8.
 14. curb-weight:              continuous from 1488 to 4066.
 15. engine-type:              dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
 16. num-of-cylinders:         eight, five, four, six, three, twelve, two.
 17. engine-size:              continuous from 61 to 326.
 18. fuel-system:              1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
 19. bore:                     continuous from 2.54 to 3.94.
 20. stroke:                   continuous from 2.07 to 4.17.
 21. compression-ratio:        continuous from 7 to 23.
 22. horsepower:               continuous from 48 to 288.
 23. peak-rpm:                 continuous from 4150 to 6600.
 24. city-mpg:                 continuous from 13 to 49.
 25. highway-mpg:              continuous from 16 to 54.
 26. price:                    continuous from 5118 to 45400.    
    
Source of original data set: 
    url = http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
 
Binary question:
     
     are cars with higher horsepower more expensive than those with lower horseporwer?
     does the horsepower provide an indication on the price of the car?
     
Non Binary question:
    (What is...? How many...? When does...?)
    
    
"""
#Importing libraries
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################
############################ Cleaning data ###################################
##############################################################################

#Read in Data from the online archive
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

auto = pd.read_csv("imports-85.data",header=None) ##This line was used to experiment with the local file 

#Column names from the original data (names come from the original data repository)
auto.columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

#Check the unique values for a particular attribute to later on compare it with itself to make sure the code is working
auto.loc[:,"symboling"].unique()

#Numeric columns in the data set
numerics = ["normalized-losses","wheel-base","length","width","height","curb-weight","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

#Categorical columns in the data set
categorical = ["symboling","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","engine-type","num-of-cylinders","fuel-system"]

#Corece to numeric and impute medians for Each Attribute
for i in numerics:
    auto.loc[:,i] = pd.to_numeric(auto.loc[:,i], errors='coerce')

#For loop to assign median values for missing numeric values
for i in numerics:
    HasNan = np.isnan(auto.loc[:,i]) #Find the terms in each column that have nan values
    Median  = np.nanmedian(auto.loc[:,i]) #Determine the median value for each numeric column
    auto.loc[HasNan, i] = Median #Replace the nan values for the mean of the column/attribute

#For loop to remove and replace Outliers for the mean of the group without the outliers
for i in numerics:
    LimitHi = np.mean(auto.loc[:,i]) + 2*np.std(auto.loc[:,i]) #Higher Limit for each of the numeric attributes
    LimitLo = np.mean(auto.loc[:,i]) - 2*np.std(auto.loc[:,i]) #Lower Limit for each of the numeric attributes
    FlagBad = (auto.loc[:,i] < LimitLo) | (auto.loc[:,i] > LimitHi) #Boolean for values outside limits
    FlagGood = ~FlagBad #Complement
    auto.loc[FlagBad,i] = np.mean(auto.loc[FlagGood,i]) #Replace outliers with the mean of the data w/o the outlier values

#Normalize numeric values (at least 1 column, but be consistent with other numeric data)
names = "_norm"
for i in numerics:
    auto[i + names] = (auto[i] - np.mean(auto[i]))/np.std(auto[i])

#Bin numeric variables (at least 1 column)
NB = 4 #Based on the histogram distribution of the variable
bounds = np.linspace(np.min(auto["highway-mpg"]), np.max(auto["highway-mpg"]), NB + 1) 

#Function to get bins 
def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= b[i-1])&(x < b[i])] = i
    
    y[x == b[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y

bx = bin(auto["highway-mpg"], bounds)

auto["highway-mpg-binned"] = bx

#Impute missing values
#Based on the unique values from the previous section it is seen that the 
#only attribute with missing categorical data is num-of-doors
auto.loc[:, "num-of-doors"].value_counts() 
MissingValue = auto.loc[:, "num-of-doors"] == "?" #Locate all the missing values with "?"
auto.loc[MissingValue, "num-of-doors"] = "four"#based on the count of the values the most common case is 4 doors

#Consolidate categorical data (at least 1 column)
auto.loc[auto.loc[:, "engine-type"] == "dohc", "engine-type"] = 1
auto.loc[auto.loc[:, "engine-type"] == "dohcv", "engine-type"] = 2
auto.loc[auto.loc[:, "engine-type"] == "l", "engine-type"] = 3
auto.loc[auto.loc[:, "engine-type"] == "ohc", "engine-type"] = 4
auto.loc[auto.loc[:, "engine-type"] == "ohcf", "engine-type"] = 5
auto.loc[auto.loc[:, "engine-type"] == "ohcv", "engine-type"] = 6
auto.loc[auto.loc[:, "engine-type"] == "rotor", "engine-type"] = 7

#normalized the categorial data
auto["engine-type" + names] = (auto["engine-type"] - np.mean(auto["engine-type"]))/np.std(auto["engine-type"]) #mean and std dev
#auto["engine-type" + names] = (auto["engine-type"] - min(auto["engine-type"]))/(max(auto["engine-type"])-min(auto["engine-type"])) #min/max
auto["symboling"] = auto["symboling"].apply(pd.to_numeric)
auto["symboling" + names] = (auto["symboling"] - np.mean(auto["symboling"]))/np.std(auto["symboling"]) #mean and std dev


##############################################################################
############################## K-Means #######################################
##############################################################################

first_data_set = auto[["horsepower_norm", "price_norm"]]#Selecting just 2 attributes from the data (the values are normalized with respect its mean and std dev)
#second_data_set = auto[["compression-ratio_norm", "horsepower_norm"]] #Selecting just 2 attributes from the data (the values are normalized with respect its mean and std dev)
second_data_set = auto[["symboling_norm", "price_norm"]] #Selecting just 2 attributes from the data (the values are normalized with respect its mean and std dev)
#second_data_set = auto[["horsepower_norm", "price_norm"]] #Selecting just 2 attributes from the data (the values are normalized with respect its mean and std dev)
third_data_set = auto[["engine-type_norm", "horsepower_norm"]] #Selecting just 2 attributes from the data (the values are normalized with respect its mean and std dev)

a = 0.8
s = 100
ec = 'black' # edge color

#Exploration of the data to determine the number of clusters that best describe the points 
#labels_data_first = []
#for i in range(2,7): # from 2 to 7
#    k_means = KMeans(n_clusters=i) #Kmeans for each particular number of clusters from 2 to 7
#    values = k_means.fit(first_data_set)
#    labels = values.labels_ #labels
#    labels_data_first.append(labels) #appending the labels to an empty list
#    center = values.cluster_centers_ #center of the cluster
#    centers = np.array(center) #transform to an numpy array to later use it on the plot
#    plt.scatter(first_data_set["wheel-base_norm"], first_data_set["horsepower_norm"], c=labels, alpha=a, s=s, edgecolors=ec) #plot the cloud of points with clear distinction of the clusters
#    plt.scatter(centers[:,0], centers[:,1], marker="*", color='r') #plot the center for each cluster
#    plt.title('N_Clusters = ' + str(i))
#    plt.xlabel("wheel-base_norm")
#    plt.ylabel("horsepower_norm")
#    plt.show()

labels_data_second = []
for i in range(2,7): # from 2 to 7
    k_means = KMeans(n_clusters=i)
    values = k_means.fit(second_data_set)
    labels = values.labels_
    labels_data_second.append(labels)
    center = values.cluster_centers_
    centers = np.array(center)
    plt.scatter(second_data_set["symboling_norm"], second_data_set["price_norm"], c=labels, alpha=a, s=s, edgecolors=ec)
    plt.scatter(centers[:,0], centers[:,1], marker="*", color='r')
    plt.title('N_Clusters = ' + str(i))
    plt.xlabel("symboling_norm")
    plt.ylabel("price_norm")
    plt.show()
    
labels_data_third = []
for i in range(2,7): # from 2 to 7
    k_means = KMeans(n_clusters=i)
    values = k_means.fit(third_data_set)
    labels = values.labels_
    labels_data_third.append(labels)
    center = values.cluster_centers_
    centers = np.array(center)
    plt.scatter(third_data_set["engine-type_norm"], third_data_set["horsepower_norm"], c=labels, alpha=a, s=s, edgecolors=ec)
    plt.scatter(centers[:,0], centers[:,1], marker="*", color='r')
    plt.title('N_Clusters = ' + str(i))
    plt.xlabel("engine-type")
    plt.ylabel("horsepower_norm")
    plt.show()
    
#based on the previous data the number of clusters for each set of data
   
first_data_set = auto[["horsepower_norm", "price_norm"]]

k_means = KMeans(n_clusters=4) #Kmeans for each particular number of clusters from 2 to 7
values = k_means.fit(first_data_set)
labels = values.labels_
auto["price_horsepower_label"] = labels  #storing the label on the dataset
center = values.cluster_centers_ #center of the cluster
centers = np.array(center) #transform to an numpy array to later use it on the plot
plt.scatter(first_data_set["horsepower_norm"], first_data_set["price_norm"], c=labels, alpha=a, s=s, edgecolors=ec) #plot the cloud of points with clear distinction of the clusters
plt.scatter(centers[:,0], centers[:,1], marker="*", color='r') #plot the center for each cluster
plt.title('N_Clusters = ' + str(3))
plt.xlabel("horsepower_norm")
plt.ylabel("price_norm")
plt.show()

k_means = KMeans(n_clusters=2)
values = k_means.fit(third_data_set)
labels = values.labels_
labels = values.labels_
auto["eng_horsepower_label"] = labels  #storing the label on the dataset
center = values.cluster_centers_ #center of the cluster
centers = np.array(center) #transform to an numpy array to later use it on the plot
plt.scatter(third_data_set["engine-type_norm"], third_data_set["horsepower_norm"], c=labels, alpha=a, s=s, edgecolors=ec)  #plot the cloud of points with clear distinction of the clusters
plt.scatter(centers[:,0], centers[:,1], marker="*", color='r') #plot the center for each cluster
plt.title('N_Clusters = ' + str(2))
plt.xlabel("engine-type")
plt.ylabel("horsepower_norm")
plt.show()