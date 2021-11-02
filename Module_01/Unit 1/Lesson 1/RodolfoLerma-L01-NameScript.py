# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:51:21 2020

@author: Rodolfo Lerma
"""
#### Name Function ####
def my_name(): # A funcion my_name is being created
    return print("Rodolfo Lerma") # There is no input needed to run the function as it will return my name
    
my_name() #The function is being called

#### Time and Date Function ####
from datetime import datetime as dt #importing the wanted/needed library datetime

def date_and_time(): # A function date_and_time() is being created
    return print (dt.now()) # The function will print the date and time using the following format YYYY-MM-DD HH:MM:SS

date_and_time() #The function is being called





