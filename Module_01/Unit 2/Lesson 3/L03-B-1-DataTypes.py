"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

""" Data Types """

# Create an integer
x = 7

# Determine the data type of x
type(x)
#################

# Add 3 to x
x + 3
#################

# Create a float
x = 7.0

# Determine the data type of x
type(x)
#################
# Add 3 to x
x + 3.0
#################

# Add a float, an integer, and a Boolean
7 + 3.0 + True
#################

# Create a string
x = "a"

# Determine the data type of x
type(x)
#################

# Add b to x
x + "b"
################

# Try to add 3 to a string
x + 3
################

# Create a string
x = "7"

# Determine the data type of x
type(x)
################

# If we try to add 3 to x, we will get an error
x + 3

# Add "3" to x
x + "3"
################
 
import numpy as np

# Create an array of integers
x = np.array([5, -7, 1, 1, 99])

# Determine the data type of x
type(x)
################

# Find out the data type for the elements in the array
x.dtype.name
################

# If we can add 3 to this array.
x + 3
################

# Create an array of strings
x = np.array(["abc", "", " ", "?", "7"])

# Determine the data type of x
type(x)

# Find out the data type for the elements in the array
x.dtype.name
################

# If we try to add 3 to this array of strings we will get a TypeError.
x + 3

#################