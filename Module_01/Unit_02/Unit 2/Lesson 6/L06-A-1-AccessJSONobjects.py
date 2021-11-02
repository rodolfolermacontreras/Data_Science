"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""
"""
Example script for JSON assessor methods
JavaScript Object Notation: text based structure for storing data
- Object (maps to a python dictionary {})
- Array (maps to a python list)

objects can be access by key
arrays can be access by index
"""

json_obj =  {
        "name": "Steven",
        "age": 27,
        "siblings": ["Anna", "Peter", "Lowell"],
        "cars": {
                "Toyota":["Tercel", "Forerunner"],
                "Nissan":["Versa", "Sentra"]
                }
        }
###############################
        
#access object by key
print(json_obj["name"])

#access list elements by index
print(json_obj["siblings"][1])

#access nested elements by key and index
print(json_obj["cars"]["Toyota"][0])

#############################