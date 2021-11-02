# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:05:57 2020

@author: Rodolfo Lerma

WebScaper to get the number of links presented in the website

"""
# Libraries needed for this assignment
import requests
from bs4 import BeautifulSoup 

#Link for the website used
url = "https://bibleproject.com/church-at-home/"

#Pull down data from website (html) and store it as an object
response = requests.get(url)

#Getting the content from the page
content = response.content

#Converting this to something a human can read
soup = BeautifulSoup(content, "lxml")

#Finding anything inside a tag (looking for links)
all_a_https = soup.find_all("a") 

#Getting elements into a python list
links = []
for x in all_a_https:
    links.append(x.get("href"))

#For loop to count any element from the list that contains the "https: for a link"
count = 0
for x in links:
    if "https:" in x:
        count +=1

#Print the number of links found in the website page
print(count)

#print the list with the values
#print(links)



