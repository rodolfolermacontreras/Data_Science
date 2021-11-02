# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:50:50 2020

@author: Rodolfo Lerma

Supervised Learning Script

General information about the "imports-85":
    After cleaning and normalizing data: 44 (originally we had 26 attributes/columns)
    Original Number of observations: 205
    The data consists of 3 types of entities:
        1) Specifications of a car 
        2) Insurance risk rating
        3) Normalized losses in use as compared to other cars
 
 Source of original data set: 
    url = http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
    
Missing values were substituted for median for Numeric attributes.
Outliers were substituted for mean of the Numeric attribute w/o outliers.
All the numeric values were normilized (using Mean and Std Dev) and added as new columns, ending with _norm
Some of the categorical variables were hot-encoded.

The expert label chosed was Price, and that variable was binned in 3 categories: 
    1 Cheap
    2 Mid Range
    3 Expensive
    
Binary question:
    Will a car with the following characteristics be label Expensive? (Yes/No):
        compression-ratio: 8.3
        horsepower: 140
        peak rpm: 5500
        city mpg: 17
        highway mpg: 20
        engine type: overhead
        drive wheels: fwd
        fuel type: gas
        body style: sedan
        make: audi

The label for this particular care says that it would be a 3, meaning: an expensive car.
And the model predicted a: [2] Mid Range

My classification model is based on the following columns:
The normalized values from: compression-ratio, horsepower, peak rpm, city mpg & highway mpg
One-hot encoding from the following colummns: engine type, drive wheels, fuel type, body style & make

"""
#Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

##############################################################################
############################ Cleaning data ###################################
##############################################################################

#Read in Data from the online archive
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

auto = pd.read_csv("imports-85.data",header=None) ##This line was used to experiment with the local file 

#Column names from the original data (names come from the original data repository)
auto.columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

#Check the unique values for a particular attribute to later on compare it with itself to make sure the code is working
auto.loc[:,"symboling"].unique()

#Numeric columns in the data set
numerics = ["normalized-losses","wheel-base","length","width","height","curb-weight","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

#Categorical columns in the data set
categorical = ["symboling","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","engine-type","num-of-cylinders","fuel-system"]

#Corece to numeric and impute medians for Each Attribute
for i in numerics:
    auto.loc[:,i] = pd.to_numeric(auto.loc[:,i], errors='coerce')

#For loop to assign median values for missing numeric values
for i in numerics:
    HasNan = np.isnan(auto.loc[:,i]) #Find the terms in each column that have nan values
    Median  = np.nanmedian(auto.loc[:,i]) #Determine the median value for each numeric column
    auto.loc[HasNan, i] = Median #Replace the nan values for the mean of the column/attribute

#For loop to remove and replace Outliers for the mean of the group without the outliers
for i in numerics:
    LimitHi = np.mean(auto.loc[:,i]) + 2*np.std(auto.loc[:,i]) #Higher Limit for each of the numeric attributes
    LimitLo = np.mean(auto.loc[:,i]) - 2*np.std(auto.loc[:,i]) #Lower Limit for each of the numeric attributes
    FlagBad = (auto.loc[:,i] < LimitLo) | (auto.loc[:,i] > LimitHi) #Boolean for values outside limits
    FlagGood = ~FlagBad #Complement
    auto.loc[FlagBad,i] = np.mean(auto.loc[FlagGood,i]) #Replace outliers with the mean of the data w/o the outlier values

#Normalize numeric values (at least 1 column, but be consistent with other numeric data)
names = "_norm"
for i in numerics:
    auto[i + names] = (auto[i] - np.mean(auto[i]))/np.std(auto[i])

#Impute missing values
#Based on the unique values from the previous section it is seen that the 
#only attribute with missing categorical data is num-of-doors
auto.loc[:, "num-of-doors"].value_counts() 
MissingValue = auto.loc[:, "num-of-doors"] == "?" #Locate all the missing values with "?"
auto.loc[MissingValue, "num-of-doors"] = "four"#based on the count of the values the most common case is 4 doors

#Consolidate categorical data (at least 1 column)
auto.loc[auto.loc[:, "engine-type"] == "dohc", "engine-type"] = "Dual_OverHead"
auto.loc[auto.loc[:, "engine-type"] == "dohcv", "engine-type"] = "Dual_OverHead"
auto.loc[auto.loc[:, "engine-type"] == "l", "engine-type"] = "L_Engine"
auto.loc[auto.loc[:, "engine-type"] == "ohc", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "ohcf", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "ohcv", "engine-type"] = "OverHead"
auto.loc[auto.loc[:, "engine-type"] == "rotor", "engine-type"] = "Rotary_Engine"

#One-hot encode categorical data
auto.loc[:, "Dual_Overhead"] = (auto.loc[:, "engine-type"] == "Dual_OverHead").astype(int)
auto.loc[:, "L_Engine"] = (auto.loc[:, "engine-type"] == "L_Engine").astype(int)
auto.loc[:, "Overhead"] = (auto.loc[:, "engine-type"] == "OverHead").astype(int)
auto.loc[:, "Rotary"] = (auto.loc[:, "engine-type"] == "Rotary_Engine").astype(int)

auto.loc[:, "4wd"] = (auto.loc[:, "drive-wheels"] == "4wd").astype(int)
auto.loc[:, "fwd"] = (auto.loc[:, "drive-wheels"] == "fwd").astype(int)
auto.loc[:, "rwd"] = (auto.loc[:, "drive-wheels"] == "rwd").astype(int)

auto.loc[:, "four"] = (auto.loc[:, "num-of-doors"] == "four").astype(int)
auto.loc[:, "two"] = (auto.loc[:, "num-of-doors"] == "two").astype(int)

auto.loc[:, "diesel"] = (auto.loc[:, "fuel-type"] == "diesel").astype(int)
auto.loc[:, "gas"] = (auto.loc[:, "fuel-type"] == "gas").astype(int)

auto.loc[:, "hardtop"] = (auto.loc[:, "body-style"] == "hardtop").astype(int)
auto.loc[:, "wagon"] = (auto.loc[:, "body-style"] == "wagon").astype(int)
auto.loc[:, "sedan"] = (auto.loc[:, "body-style"] == "sedan").astype(int)
auto.loc[:, "hatchback"] = (auto.loc[:, "body-style"] == "hatchback").astype(int)
auto.loc[:, "convertible"] = (auto.loc[:, "body-style"] == "convertible").astype(int)

#In this case since there were to many names a For loop was used to do a One-Hot enconding for this attribute
for i in auto.make.unique():
    auto.loc[:,i] = (auto.loc[:, "make"] == i).astype(int)

#Remove obsolete columns
auto = auto.drop("engine-type", axis=1)
auto = auto.drop("drive-wheels", axis=1)
auto = auto.drop("num-of-doors", axis=1)
auto = auto.drop("fuel-type", axis=1)
auto = auto.drop("body-style", axis=1)
auto = auto.drop("make",axis=1)

#This section is to bin the Price into 3 Categories that in this case as an example will be named:
# 1 - Cheap, 2 - Mid range & 3 - Expensive
NB = 2
bounds = np.linspace(np.min(auto["price"]), np.max(auto["price"]), NB + 1)

def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= b[i-1])&(x < b[i])] = i
    
    y[x == b[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y

#This is a new attribute with the price variable binned into 3 categories
# 1 - Cheap, 2 - Mid range & 3 - Expensive (this is an oversimplification)    
auto["price_binned"] = bin(auto["price"], bounds) 

##############################################################################
############################## K-Means #######################################
##############################################################################

#attributes_numeric = ["wheel-base_norm","length_norm","width_norm","height_norm","curb-weight_norm","bore_norm","stroke_norm","compression-ratio_norm","horsepower_norm","peak-rpm_norm","city-mpg_norm","highway-mpg_norm"]
attributes_numeric = ["compression-ratio_norm","horsepower_norm","peak-rpm_norm","city-mpg_norm","highway-mpg_norm"]
attributes_categorical = ["Dual_Overhead","L_Engine","Overhead","Rotary","4wd","fwd","rwd","four","two","diesel","gas","hardtop","wagon","sedan","hatchback","convertible", "alfa-romero", "audi", "bmw", "chevrolet", "dodge", "honda", "isuzu", "jaguar", "mazda", "mercedes-benz", "mercury","mitsubishi", "nissan", "peugot", "plymouth", "porsche", "renault", "saab", "subaru", "toyota", "volkswagen", "volvo"]
attributes = attributes_numeric + attributes_categorical

#Removing just one particular row (case) as an example. That row was stored before removal as "particular_case"
particular_case = auto.iloc[8]

'''
compression-ratio: 8.3
horsepower: 140
peak rpm: 5500
city mpg: 17
highway mpg: 20
engine type: overhead
drive wheels: fwd
fuel type: gas
body style: sedan
make: audi
'''
#Dropping that row from the data set
auto = auto.drop([8])

#Split the data
X = auto[attributes]
y = auto["price_binned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#Training the Model
#K Nearest Neighbors classifier
print ('\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X_train, y_train.astype('int'))

#Predictions and probabilities
proba = knn.predict_proba(X_test)[:,1]
prediction = knn.predict(X_test)

print ("\nPredictions for test set:\n")
print (prediction)
print ('\nActual class values:\n')
print (y_test.tolist())

#Accuracy rate
#number of correct predictions divided by the total number of predictions
z = len(y_test) #Number of predictions
actual_predictions = y_test == knn.predict(X_test)
accuracy_rate = sum(actual_predictions)/z
print("\nAccuracy rate for this model (%):\n")
print(accuracy_rate * 100)

#Creating a Data Frame for the Test, Prediction and Probaility Data
y_test2 = y_test.to_numpy()
y_test2 = y_test2.tolist()
prediction2 = prediction.tolist()
proba2 = proba.tolist()
results = np.array([y_test2, prediction2, proba2])
knn_data = pd.DataFrame(data=results, index=['Test_Values', 'Prediction', 'Probability'])
knn_auto = knn_data.T

#Store the file as CSV file
knn_auto.to_csv('supervised_knn_auto_data.csv', sep=",", index=False)

#Particular question:
particular_case = particular_case[attributes]
results_particular_case = knn.predict([particular_case])
print('\nThe Prediction for the price of the Audi Car is:\n')
print(results_particular_case)