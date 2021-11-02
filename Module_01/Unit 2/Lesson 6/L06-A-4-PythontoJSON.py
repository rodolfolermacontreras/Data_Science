"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import json 

""" Python Dictionary to JSON """
# create a python dictionary
d= {}
d["Name"] = "Steve"
d["Country"] = "Merica"

# How many elements are in d? Check the Variable explorer.
#########################

# convert python dictionary to JSON using json.dumps()
json_obj = json.dumps(d)
# How many elements are in json_obj?
#########################

# view JSON objectp
print(json_obj)

#########################
""" Python List to JSON """
# create a python array
py_list = ["abc", 123]
# How many elements are in py_list?
#########################

# convert python list to JSON array using json.dumps()
json_array = json.dumps(py_list)
# How many elements are in json_array?
########################

# view list structure and data type
print(py_list)
print(type(py_list))

print(json_array)
print(type(json_array)) 
# note that JSON arrays are strings
########################