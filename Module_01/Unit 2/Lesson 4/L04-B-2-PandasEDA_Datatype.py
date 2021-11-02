"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# pandas EDA and datatypes
# Why are datatypes important in EDA?

# Import required packages
import pandas as pd
import matplotlib.pyplot as plt
####################

# Read in data as a pandas data frame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
Auto = pd.read_csv(url,sep='\s+', header=None)
Auto.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
               'acceleration', 'model year', 'origin', 'car name']
####################

# Histogram of horsepower
plt.hist(Auto.loc[:,'horsepower'])
####################

# Something is wrong with the histogram

# Present sorted values
Auto.loc[:,'horsepower'].sort_values()
####################

# We determine the locations of the question marks
QuestionMark = Auto.loc[:,'horsepower'] == "?"
# How many question marks?
sum(QuestionMark)
####################

# We remove the rows with question marks
Auto = Auto.loc[~QuestionMark, :]
# Verify that question marks are gone
sum(Auto.loc[:,'horsepower'] == "?")
####################

# Histogram of horsepower
plt.hist(Auto.loc[:,'horsepower'])
####################

# See sorted values again
Auto.loc[:,'horsepower'].sort_values()
####################

Auto.dtypes
####################

# Cast the type
Auto.loc[:,'horsepower'] = Auto.loc[:,'horsepower'].astype(float)

# Let's check if type changed
Auto.dtypes
####################

# Let's try to get a proper histogram one more time
plt.hist(Auto.loc[:,'horsepower'])
####################

# Your turn:
# Look at the types of the columns in the following data frame!
# Create a histogram of the age column in the mammographic-masses dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
Mamm.loc[:,'Age'].sort_values()
QuestionMark1 = Mamm.loc[:,'Age'] == "?"
Mamm = Mamm.loc[~QuestionMark1, :]
Mamm.loc[:,'Age'].sort_values()
Mamm.loc[:,'Age'] = Mamm.loc[:,'Age'].astype(float)
plt.hist(Mamm.loc[:,'Age'])

Mamm.dtypes

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url, header=None)
# http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
Adult_columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
Adult.columns = Adult_columns + ["Income"]
Adult.dtypes

