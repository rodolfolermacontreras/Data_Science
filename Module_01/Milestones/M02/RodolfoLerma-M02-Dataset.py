# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:36:20 2020

@author: Rodolfo Lerma

Milestone 02
Clean dataset 

Prepare a data set with known missing values:

    Account for aberrant data (missing and outlier values).
    Normalize numeric values (at least 1 column).
    Bin categorical variables (at least 1 column).
    Construct new categorical variables.
    Remove obsolete columns.

"""

# import package
import pandas as pd
import numpy as np

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

#Decode categorical data
#To get a status of the kind of values we have for each categorical variable
for i in categorical:
    x = auto.loc[:,i].unique()
    print (x)

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
print ("\n\nBinned variable x, for ", NB, "bins\n")
print ("Bin boundaries: ", bounds)
print ("Binned variable: ", bx)

auto["highway-mpg-binned"] = bx
    
#Decode Symboling
Replace = auto.loc[:, "symboling"] == "3"
auto.loc[Replace, "symboling"] = "Very_risky"

Replace = auto.loc[:, "symboling"] == "2"
auto.loc[Replace, "symboling"] = "Somehow_risky"

Replace = auto.loc[:, "symboling"] == "1"
auto.loc[Replace, "symboling"] = "Low_risk"

Replace = auto.loc[:, "symboling"] == "0"
auto.loc[Replace, "symboling"] = "Neutral"

Replace = auto.loc[:, "symboling"] == "-1"
auto.loc[Replace, "symboling"] = "Safe"

Replace = auto.loc[:, "symboling"] == "-2"
auto.loc[Replace, "symboling"] = "Very_Safe"

#Impute missing values
#Based on the unique values from the previous section it is seen that the 
#only attribute with missing categorical data is num-of-doors
auto.loc[:, "num-of-doors"].value_counts() 
MissingValue = auto.loc[:, "num-of-doors"] == "?" #Locate all the missing values with "?"
auto.loc[MissingValue, "num-of-doors"] = "four"#based on the count of the values the most common case is 4 doors

#Consolidate categorical data (at least 1 column)
auto.loc[auto.loc[:, "engine-type"] == "dohc", "engine-type"] = "Dual_OverHead"
auto.loc[auto.loc[:, "engine-type"] == "dohcv", "engine-type"] = "Dual_OverHead"
auto.loc[auto.loc[:, "engine-type"] == "l", "engine-type"] = "L_Engine"
auto.loc[auto.loc[:, "engine-type"] == "ohc", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "ohcf", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "ohcv", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "rotor", "engine-type"] = "Rotary_Engine"

#One-hot encode categorical data with at least 3 categories (at least 1 column)
auto.loc[:, "Dual_Overhead"] = (auto.loc[:, "engine-type"] == "Dual_OverHead").astype(int)
auto.loc[:, "L_Engine"] = (auto.loc[:, "engine-type"] == "L_Engine").astype(int)
auto.loc[:, "Overhead"] = (auto.loc[:, "engine-type"] == "OverHead").astype(int)
auto.loc[:, "Rotary"] = (auto.loc[:, "engine-type"] == "Rotary_Engine").astype(int)

#Remove obsolete columns
auto = auto.drop("engine-type", axis=1)

#Clean Data
auto.to_csv('clean_auto_data.csv', sep=",", index=False)