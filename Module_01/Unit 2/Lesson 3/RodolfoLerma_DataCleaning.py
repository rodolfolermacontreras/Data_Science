# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:20:51 2020

@author: Rodolfo Lerma
This code contains 3 functions to clean data by either:
    1) Removing data (outliers)
    2) Replacing data (outliers) for the mean of the group without outliers
    3) Replacing missing data for the median of the group after removing those missing values
"""

""" Data Cleaning """

# Import the Numpy library
import numpy as np

#### Create arr1 #####
np.random.seed(30) # Seed to be able to obtain the same random numbers every time
x = np.random.randint(1,21,30) # This obtains an array of 30 values ranging from 1 to 20 as type int

# 5 values that would be outliers compared to the 30 original values
outliers = np.array([100,70])

# Creating the arr1 with x values (1-20) and Outliers
# by appending the two np.arrays
arr1 = np.append(x,outliers)

#### Create arr2 #####
# 4 values as improper non-numeric missing values
nonnumeric = np.array(["?"," ","nan"])

# Creating the arr2 with x values (1-20) and nonnumeric by appending the two np.arrays
arr2 = np.append(x,nonnumeric)

print(arr1) # Print arr1 to have as reference
print(arr2) # Print arr2 to have as reference

def remove_outliers(x):
    LimitHi = np.mean(x) + 2*np.std(x) # The high limit for acceptable values is the mean plus 2 standard deviations    
    LimitLo = np.mean(x) - 2*np.std(x) # The low limit for acceptable values is the mean plus 2 standard deviations
    FlagGood = (x >= LimitLo) & (x <= LimitHi) # Create Flag for values within limits 
    x = x[FlagGood]# Assigning the values without outliers to the original variable
    return x # Returing the values without outliers

def replace_outliers(x):
    LimitHi = np.mean(x) + 2*np.std(x) # The high limit for acceptable values is the mean plus 2 standard deviations    
    LimitLo = np.mean(x) - 2*np.std(x) # The low limit for acceptable values is the mean plus 2 standard deviations
    FlagGood = (x >= LimitLo) & (x <= LimitHi) # Create Flag for values within limits 
    FlagBad = ~FlagGood # FlagGood is the complement of FlagGood
    x[FlagBad] = np.mean(x[FlagGood]) # Replace outleirs with the mean of non-outliers
    z = x # Assigning this local value to a new variable to avoid confusion with the global variable 
    return z # Returing the original array but replacing the outliers with the mean of the rest of the values

def fill_median(x):
    FlagGood = (x != "?") & (x != " ") & (x != "nan") # Do not allow specific texts
    FlagBad = ~FlagGood # FlagGood is the complement of FlagGood
    x[FlagBad] = np.median(x[FlagGood].astype('float64'))
    y = x # Assigning this local value to a new variable to avoid confusion with the global variable
    return y # Returning the original array but replacing the missing values with the median of the rest of the values

# Calling the function to remove outliers and assigning the value to arr1clean
arr1clean = remove_outliers(arr1)

# Calling the function to replace outliers with the mean and assigning the value to arr1replace
arr1replace = replace_outliers(arr1)

# Calling the function to replace missing values with the median and assigning the value to arr2fill
arr2fill = fill_median(arr2)

#As a verification of the values print the results
print(arr1clean)
print(arr1replace)
print(arr2fill)

