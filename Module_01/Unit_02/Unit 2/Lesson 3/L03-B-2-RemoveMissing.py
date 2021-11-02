"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

""" Missing Values -- Removal """

# Import NumPy
import numpy as np

# Create an array
x = np.array([2, 1, " ", 1, 99, 1, 5, 3, "?", 1, 4, 3])
################

# Attempt to tally values that are larger than 4
sum(x > 4)

# Find out the data type for x:
type(x)

# Find out the data type for the elements in the array
x.dtype.name
#################

# Do not allow specific texts
FlagGood = (x != "?") & (x != " ")

# Find elements that are numbers
FlagGood = [element.isdigit() for element in x]
##################

# Select only the values that look like numbers
x = x[FlagGood]

x
##################
 
# Attempt to tally values that are larger than 4
sum(x > 4)
##################

# Need to cast the numbers from text (string) to real numeric values
x = x.astype(int)

x
##################

# tally values that are larger than 4
sum(x > 4)

c = "7"
2 + int(c)
#################

import numpy as np
x = np.array([5, -7, 1.1, 1, 99])
x.dtype.name

import numpy as np
x = np.array([-2, 1, "", 1, 20, 1, 5, -2, "X", -1, 4, 3])
FlagGood = (x != "") & (x != "X")
np.mean(x[FlagGood].astype(str))

x = np.array([5, -7, 1.1, 1, 2])
Hi = np.mean(x) + 2*np.std(x)
Lo = np.mean(x) - 2*np.std(x)

Hi = np.mean(x) + 2*np.std(x)
Lo = np.mean(x) - 2*np.std(x)
Good = (x < Hi) & (x > Lo)
y = x[~Good]

c = "7"
int("2" + c)

c = 7
int("2" + c)

c = "7"
int(2*c)

import numpy as np
a = 5.1
b = np.array([3]) + a

b.dtype

a = 5
b = a + 3.1

import numpy as np
x = np.array([5, -7, 1, 1, 99])
x.dtype.name

import numpy as np
x = np.array([-2, 1, "", 1, 20, 1, 5, -2, "X", -1, 4, 3])
FlagGood = (x != "") & (x != "X")

np.mean(x[FlagGood].astype(str))

c = "7"
int(2 + c)