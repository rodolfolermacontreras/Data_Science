# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:19:14 2020

@author: Rodolfo Lerma

Exploratory Data Analysis Script

The Data Set used in this assignment contains 20 attributes. Out of those 20 just 
6 are for numeric values. 
The number of outliers per attribute is summarized below:
    Age: 6
    Bilirubin: 10
    Alk: 7
    Sgot: 6
    Albumin: 9
    Protime: 14
These outliers were identified by looking for any value that was above or below
the mean value by 2 Standard Deviations.
The attributes that required of missing values are the following:
    Age: 0
    Bilirubin: 6
    Alk: 29
    Sgot: 4
    Albumin: 16
    Protime: 67
This numeric attributes contained the symbol "?" for missing value.
All the numeric attributes were plotted to see what kind of distribution the 
data was showing.
No rows were removed at this time since for the numeric attributes a replacement was
perfomed for both outliers and missing values. Potentially the attribute "Protime"
could be removed as it contains 67 missing entries, which is close to 40% of the 
entire data set.
For the non numeric attributes potentially later on some rows will be remove 
since missing values are present and cannot (and should not) be guess.
"""

#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
#Read in Data from the online archive
#hepa = pd.read_csv("hepatitis.data",header=None) 
hepa = pd.read_csv(url,header=None) 

#Column names from the original data (names come from the original data repository)
hepa.columns = ["Class","Age","Sex","Steroid","Antivirals","Fatigue","Malaise","Anorexia","Liver Big","Liver Firm","Spleen Palpable","Spiders","Ascites","Varices","Bilirubin","Alk","Sgot","Albumin","Protime","Histology"]

#Check the Data type for each attribute
hepa.dtypes

#Check the shape of the data array
hepa.shape

#Check the unique values for a particular attribute to later on compare it with itself to make sure the code is working
hepa.loc[:,"Age"].unique()

#Numeric columns in the data set
numerics = ["Age","Bilirubin","Alk","Sgot","Albumin","Protime"]

#Corece to numeric and impute medians for Each Attribute
for i in numerics:
    hepa.loc[:,i] = pd.to_numeric(hepa.loc[:,i], errors='coerce')

#For loop to assign median values for missing numeric values
for i in numerics:
    HasNan = np.isnan(hepa.loc[:,i]) #Find the terms in each column that have nan values
    Median  = np.nanmedian(hepa.loc[:,i]) #Determine the median value for each numeric column
    hepa.loc[HasNan, i] = Median #Replace the nan values for the mean of the column/attribute

#For loop to find outliers and replace them with the mean of the data w/o the outlier values
for i in numerics:
    LimitHi = np.mean(hepa.loc[:,i]) + 2*np.std(hepa.loc[:,i]) #Higher Limit for each of the numeric attributes
    LimitLo = np.mean(hepa.loc[:,i]) - 2*np.std(hepa.loc[:,i]) #Lower Limit for each of the numeric attributes
    FlagBad = (hepa.loc[:,i] < LimitLo) | (hepa.loc[:,i] > LimitHi) #Boolean for values outside limits
    #x = sum(FlagBad) #line that could be printed to get the number of outliers on each attribute
    FlagGood = ~FlagBad #Complement
    hepa.loc[FlagBad,i] = np.mean(hepa.loc[FlagGood,i]) #Replace outliers with the mean of the data w/o the outlier values

#For loop to plot the histograms for each of the numeric attributes in the hepatitis data set
for i in numerics:
    plt.hist(hepa.loc[:,i])
    plt.xlabel(i)
    plt.ylabel("Frequency")
    plt.show()

#Scatterplot for the entire data set   
scatter_matrix(hepa,figsize=[20,20], s=1000) 
plt.show()

#Std Dev for each of the numeric attributes in the hepatitis data set
for i in numerics:
    std = np.std(hepa[i])
    print("Std Dev for " + i + " is:")
    print(std)
    print('')