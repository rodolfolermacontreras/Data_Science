"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import json 
""" Pretty Print JSON """
# create JSON object
json_data = '{"name":"Steven", "city":"Seattle"}'
# How many elements in json_data? What type?
#########################

# convert to python
python_obj = json.loads(json_data)
# How many elements in python_obj? What type?
#########################

# print python dictionary before prettiness
print(json.dumps(python_obj))

# print using sorted keys and indentation
print(json.dumps(python_obj, sort_keys=True, indent=4))

#######################
""" Writing JSON to a File """

# create an empty python dictionary
data = {}

# create list object to append entries to dict
data['people'] = []  
# How many elements in data?
#######################

# append entries
data['people'].append({'name': 'Steven', 
   'website': 'uw.edu', 
   'city': 'Seattle'})
data['people'].append({'name': 'Annie', 
    'website': 'ford.com', 
    'city': 'Detroit'})
#########################

#print(json.dumps(python_obj, sort_keys=True, indent=4))

#########################

# use json.dump() to convert and write to file
with open("test_file.txt", 'w') as outfile:
    json.dump(data, outfile)
#########################
    
# Where is the test_file.txt located?
import os
os.getcwd()
#########################
