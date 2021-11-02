# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:36:22 2020

@author: Rodolfo Lerma

Category Data Script

General information about the "imports-85":
    Number of attributes: 26.
    The data consists of 3 types of entities:
        1) Specifications of a car 
        2) Insurance risk rating
        3) Normalized losses in use as compared to other cars

The data was split into 2 main categories (see code below): 
numerics (for numerical data) and categorical
The "symboling" attribute was decoded, this because the variable corresponds
to the degree to which the auto is more (or less) risky than its price indicates
+3 indicates a high risk factor and -2 pretty safe risk factor.
The "num-of-doors" attribute was imputed. The variable "num-of-doors" had a couple
of missing values("?") that were changed for the categorical value that was more 
frequently seen in the variable. All the other categorical attributes did not 
have missing values.
The "engine-type" was consolidated to only 4 categories and later on the code
these 4 categories were used to create dummy columns (one hot encoded). Therefore 
the "engine-type" column was deleted.
All the categorical data was plotted this was done to have a general idea of the
distribution of the categorical attributes. The only one that was no included 
in the plots was the "engine-type" variable that transform as one-hot encoded
"""
# import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#Read in Data from the online archive
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

#auto = pd.read_csv("imports-85.data",header=None) ##This line was used to experiment with the local file 
auto = pd.read_csv(url,header=None) 

#Column names from the original data (names come from the original data repository)
auto.columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

#Check the Data type for each attribute
auto.dtypes

#Check the shape of the data array
auto.shape

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

#Histogram for each of the numeric attributes (this is to get an idea of the distribution of the variables)
for i in numerics:
    plt.hist(auto.loc[:,i])
    plt.xlabel(i)
    plt.ylabel("Frequency")
    plt.show()
    
#### 1 Normalize numeric values (at least 1 column, but be consistent with other numeric data)
names = "_norm"
for i in numerics:
    auto[i + names] = (auto[i] - np.mean(auto[i]))/np.std(auto[i])

#### 2 Bin numeric variables (at least 1 column)
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
print ("\n\nBinned variable x, for ", NB, "bins\n")
print ("Bin boundaries: ", bounds)
print ("Binned variable: ", bx)

#### 3 Decode categorical data
#To get a status of the kind of values we have for each categorical variable
for i in categorical:
    x = auto.loc[:,i].unique()
    print (x)
    
# Decode Symboling
Replace = auto.loc[:, "symboling"] == "3"
auto.loc[Replace, "symboling"] = "Very risky"

Replace = auto.loc[:, "symboling"] == "2"
auto.loc[Replace, "symboling"] = "Somehow risky"

Replace = auto.loc[:, "symboling"] == "1"
auto.loc[Replace, "symboling"] = "Low risk"

Replace = auto.loc[:, "symboling"] == "0"
auto.loc[Replace, "symboling"] = "Neutral"

Replace = auto.loc[:, "symboling"] == "-1"
auto.loc[Replace, "symboling"] = "Safe"

Replace = auto.loc[:, "symboling"] == "-2"
auto.loc[Replace, "symboling"] = "Very Safe"

# Impute missing values
# Based on the unique values from the previous section it is seen that the 
# only attribute with missing categorical data is num-of-doors
auto.loc[:, "num-of-doors"].value_counts() 
MissingValue = auto.loc[:, "num-of-doors"] == "?" #Locate all the missing values with "?"
auto.loc[MissingValue, "num-of-doors"] = "four"#based on the count of the values the most common case is 4 doors

#### 5 Consolidate categorical data (at least 1 column)
auto.loc[auto.loc[:, "engine-type"] == "dohc", "engine-type"] = "Dual_OverHead"
auto.loc[auto.loc[:, "engine-type"] == "dohcv", "engine-type"] = "Dual_OverHead"
auto.loc[auto.loc[:, "engine-type"] == "l", "engine-type"] = "L_Engine"
auto.loc[auto.loc[:, "engine-type"] == "ohc", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "ohcf", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "ohcv", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "rotor", "engine-type"] = "Rotary_Engine"

#### 6 One-hot encode categorical data with at least 3 categories (at least 1 column)
auto.loc[:, "Dual"] = (auto.loc[:, "engine-type"] == "Dual_OverHead").astype(int)
auto.loc[:, "L"] = (auto.loc[:, "engine-type"] == "L_Engine").astype(int)
auto.loc[:, "Overhead"] = (auto.loc[:, "engine-type"] == "OverHead").astype(int)
auto.loc[:, "Rotary"] = (auto.loc[:, "engine-type"] == "Rotary_Engine").astype(int)

#### 7 Remove obsolete columns
auto = auto.drop("engine-type", axis=1)

#### 8 Present plots for 1 or 2 categorical columns.
#Histogram for each of the numeric attributes (this is to get an idea of the distribution of the variables)
categorical2 = ["symboling","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","num-of-cylinders","fuel-system"]

#potting all the categorical attributes to get an idea of the distribution of the variables
for i in categorical2:
    auto.loc[:,i].value_counts().plot(kind='bar')
    plt.xlabel(i)
    plt.ylabel("Frequency")
    plt.show()

#Scatter plot to see the distribution of the numerical variables (not required on this assignment, but I just wanted to take a look)
#scatter_matrix(auto[numerics],figsize=[20,20], s=1000) 
#plt.show()
