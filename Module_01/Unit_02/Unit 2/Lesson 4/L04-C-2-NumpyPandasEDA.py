# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:03:52 2020

The following data frame has 13 attributes. The first step after naming columns
replaces all of the missing values with null. The "name" attribute is the only one 
that is inteded to be an objects and is left as such. After this each column was
checked with the .unique() function for any values out of the ordinary. Then I 
viewed each histogram individualy for outliers. The "mult" attribute had a few 
values that were well off the gaussian type distribution shown. Those values that
were outside two standard deviations were then replaced with the median values.

@author: John Magradey

Add a summary comment block on how the numeric variables have been treated: 
Which attributes had outliers and how were the outliers defined?
Which attributes required imputation of missing values and why?
Which attributes were histogrammed and why? 
Which attributes were removed and why? 
How did you determine which rows should be removed?
"""
#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

url="http://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data"

#Read in Data from the online archive
EcoG = pd.read_csv(url,error_bad_lines=False,header=None) 
#Assign Column names from the origional data
EcoG.columns = ["Survival","Still-Alive","Age-at-Heart-Attack","Pericardial-Effusion","Fractional-Shortening","Epss","Lvdd","Wall-Motion-Score","Wall-Motion-Index","mult","name","group","alive-at-1"]

#Check the Data type for each attribute
EcoG.dtypes

#Used to investigate the different values stored in each attribute
EcoG.loc[:,"Survival"].unique()

# Corece to numeric and impute medians for Each Attribute
EcoG.loc[:, "Survival"] = pd.to_numeric(EcoG.loc[:, "Survival"], errors='coerce')
EcoG.loc[:, "Still-Alive"] = pd.to_numeric(EcoG.loc[:, "Still-Alive"], errors='coerce')
EcoG.loc[:, "Age-at-Heart-Attack"] = pd.to_numeric(EcoG.loc[:, "Age-at-Heart-Attack"], errors='coerce')
EcoG.loc[:, "Pericardial-Effusion"] = pd.to_numeric(EcoG.loc[:, "Pericardial-Effusion"], errors='coerce')
EcoG.loc[:, "Fractional-Shortening"] = pd.to_numeric(EcoG.loc[:, "Fractional-Shortening"], errors='coerce')
EcoG.loc[:, "Epss"] = pd.to_numeric(EcoG.loc[:, "Epss"], errors='coerce')
EcoG.loc[:, "Lvdd"] = pd.to_numeric(EcoG.loc[:, "Lvdd"], errors='coerce')
EcoG.loc[:, "Wall-Motion-Score"] = pd.to_numeric(EcoG.loc[:, "Wall-Motion-Score"], errors='coerce')
EcoG.loc[:, "Wall-Motion-Index"] = pd.to_numeric(EcoG.loc[:, "Wall-Motion-Index"], errors='coerce')
EcoG.loc[:, "mult"] = pd.to_numeric(EcoG.loc[:, "mult"], errors='coerce')
EcoG.loc[:, "group"] = pd.to_numeric(EcoG.loc[:, "group"], errors='coerce')
EcoG.loc[:, "alive-at-1"] = pd.to_numeric(EcoG.loc[:, "alive-at-1"], errors='coerce')

# Loop through all of the attributes except for the pure string attribute
# If there are null values then replace those with the median value for that data
for i in range (0,EcoG.shape[1]-1):
   if i!=10:     
        HasNan= np.isnan(EcoG.iloc[:,i])
        if HasNan.sum()>0:
            EcoG.loc[HasNan,EcoG.columns.values[i]] = np.nanmedian(EcoG.iloc[:,i])



#calculates the upper and lower limits
LimitHi = np.mean(EcoG.loc[:,"mult"]) + 2*np.std(EcoG.loc[:,"mult"])
LimitLo = np.mean(EcoG.loc[:,"mult"]) - 2*np.std(EcoG.loc[:,"mult"])
#creates a boolean for values that are outside limits
FlagBad = (EcoG.loc[:,"mult"] < LimitLo) | (EcoG.loc[:,"mult"] > LimitHi)
# FlagGood is the complement of FlagBad
FlagGood = ~FlagBad
# Replace outleiers with the mean of non-outliers
EcoG.loc[FlagBad,"mult"] = np.mean(EcoG.loc[FlagGood,"mult"])


# Check the distribution of the a column
plt.hist(EcoG.loc[:, "Survival"])
plt.xlabel("Survival")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Still-Alive"])
plt.xlabel("Still-Alive")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Age-at-Heart-Attack"])
plt.xlabel("Age-at-Heart-Attack")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Pericardial-Effusion"])
plt.xlabel("Pericardial-Effusion")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Fractional-Shortening"])
plt.xlabel("Fractional-Shortening")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Epss"])
plt.xlabel("Epss")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Lvdd"])
plt.xlabel("Lvdd")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Wall-Motion-Score"])
plt.xlabel("Wall-Motion-Score")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "Wall-Motion-Index"])
plt.xlabel("Wall-Motion-Index")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "mult"])
plt.xlabel("mult")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "group"])
plt.xlabel("group")
plt.ylabel("Frequency")
plt.show()
plt.hist(EcoG.loc[:, "alive-at-1"])
plt.xlabel("alive-at-1")
plt.ylabel("Frequency")
plt.show()

# Create a scatter plot of all values
scatter_matrix(EcoG,figsize=[20,20], s=1000) 
plt.show()

#Loop through the numerica attributes and display the name and std deviation
for i in range (0,EcoG.shape[1]-1):
   if i!=10:     
       std=np.std(EcoG.iloc[:,i])
       print(EcoG.columns.values[i],"std")
       print(std)
       