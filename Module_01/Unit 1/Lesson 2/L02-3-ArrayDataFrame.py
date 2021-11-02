"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# DataStructures (imported, multi-dimensional)

import numpy as np # make numpy package usable
import pandas as pd # make pandas package usable

# Create array of ages
ageList = [9, 9, 10, 8, 12, 10, 7, 8, 10, 8, 0, 9, 7, 9, 10, 9, 6, 9, 9, 11]
ages = np.array(ageList)
ages.std()
####################

# Cannot assign a string to a numeric array
ages[0] = 'a'
####################

# Create a histogram
import matplotlib.pyplot
matplotlib.pyplot.hist(ages)
####################

Students = pd.DataFrame()
Students

# Add columns to the data frame (table)
Students['ages'] = ages
Students['Grade'] = [4, 3, 4, 3, 6, 5, 2, 3, 5, 3, 1, 4, 2, 5, 6, 4, 1, 4, 2, 6]
Students
####################

# Determine the correlation coefficient between the two columns
Students.corr()
####################
