"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""
# DataStructures (built-in, multi-dimensional)

# Documentation on lists and other data structures
# https://docs.python.org/3/tutorial/datastructures.html
####################

list
####################

City = ['USA', 704] # List can have mixed types
type(City)
####################

City.append('Seattle')
City
####################

City[1]
####################

# Equal-length lists in a list create a rectangular strucure
Name = ['Seattle', 'San Jose', 'San Jose', 'La Paz', 'La Paz']
Country = ['USA', 'USA', 'Costa Rica', 'Mexico', 'Bolivia']
Population = [704, 1030, 333, 265, 757]
Cities = [Name, Country, Population]
Cities
####################

Cities[2][3]
####################

CityNamesByCountry = {'USA':['Seattle', 'San Jose'], 'Costa Rica':'San Jose', 'Mexico':'Oaxaca'}
type(CityNamesByCountry) # dict
####################

CityNamesByCountry
####################

CityNamesByCountry['USA']
####################

CityNamesByCountry['Mexico'] = 'La Paz'
CityNamesByCountry['Bolivia'] = 'La Paz'
CityNamesByCountry
####################

list(CityNamesByCountry.keys())
####################

list(CityNamesByCountry.values())
####################

list(CityNamesByCountry.items())
####################

'CostaRica' in CityNamesByCountry
####################

'Costa Rica' in CityNamesByCountry
####################

'LaPaz' in CityNamesByCountry.values()
####################

'La Paz' in CityNamesByCountry.values()
####################

CountryByCityNames = {'Seattle':'USA', 'San Jose':['USA', 'Costa Rica'], 'La Paz':['Bolivia', 'Mexico']}
CountryByCityNames
####################

# Verify that the same information is here
Cities
