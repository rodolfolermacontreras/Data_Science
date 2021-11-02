"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# Enter the following into the IPython console and press Enter
2 + 3
########################

# Highlight the following line and use CTRL+Enter to run the code.
5 + 7
########################

# Highlight the next 2 lines of code by selecting the line numbers and then press CTRL+Enter
import numpy as np
np.exp(-1**2 / 2) / np.sqrt(2 * np.pi)

########################

# Select the next 2 lines and press CTRL+Enter. What is displayed?
import os
os.getcwd()

########################

# Assign a value to an object
x = 2

# In computer science data types are very important, because the data type
# determines what you can do
# What type of variable is x?
type(x)

#########################
# Run the next 3 lines together.
y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
print(x)
print(y)

#########################
# Run each line one at a time.
# I want x and y to be many values not just a single value
x = np.array([-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2])
# How many values are in x? Look at the variable explorer.

# In computer languages types are very important
# What type of variable is x now?
type(x)

# What kind of data is in x?
x.dtype

y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

#What type of variable is y?
type(y)

# what kind of data is in y?
y.dtype

# Run the next 2 lines together.
print(x)
print(y)
##########################
# Let's create a first graph.
# Run the next 2 lines together.
import matplotlib.pyplot as plt
plt.plot(x, y)

# Run each line one at a time.
x = np.arange(start=-2.5, stop=2.6, step=0.1)
type(x)

x.dtype

y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

plt.plot(x, y)
plt.show()
###########################
# How many values are in x? Write a command to show the values.
