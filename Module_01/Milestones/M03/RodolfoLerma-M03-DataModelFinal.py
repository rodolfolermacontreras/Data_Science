# -*- coding: utf-8 -*-
"""
Created on Mon Dec 8 12:54:25 2020

@author: Rodolfo Lerma

Milestone 03: Data Analysis

--------------------------- Recap of data used --------------------------------

Original Number of observations: 205
The data consists of 3 types of entities:
    1) Specifications of a car 
    2) Insurance risk rating
    3) Normalized losses in use as compared to other cars
 
 Source of original data set: 
    url = http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
    
The Cleaning of the data was done by:
    Missing values were substituted for median for Numeric attributes.
    Outliers were substituted for mean of the Numeric attribute w/o outliers.
    All the numeric values were normilized (using Mean and Std Dev) and added as new columns, ending with _norm
    Some of the categorical variables were hot-encoded.

The expert label chosed was Price, and that variable was binned (and binarized) in 2 categories: 
    0 Cheap
    1 Expensive
 
The variables used for this model are:
    compression-ratio (continuous)
    horsepower (continuous)
    peak rpm (continuous)
    city mpg (continuous)
    highway mpg (continuous)
    engine type (one hot encode)
    drive wheels (one hot encode)
    fuel type (one hot encode)
    body style (one hot encode)
    make (one hot encode)

The 4 Classifiers used in this analysis are:
    K Nearest Neighbors
    Naive Bayes
    Decision Tree
    Random Forest

------------------------ Results and Conclusions ------------------------------

The results as they are after running the code several times:
    ROC (evaluating AUC): 1) Random Forest, 2) K Nearest Neighbors, 3) Decision Tree & 4) Naive Bayes
    Accuracy Rate: 1)RF, 2)KNN, 3)DT & 4)NB
    Error Rate: Opposite to Accuracy
    Precision: 1)RF, 2)KNN, 3)DT, 4)NB
    Recall: 1)KNN, 2)RF, 3)NB & 4)DT
    F1 Score: 1)KNN, 2)RF, 3)DT & 4)NB

Overall the Random Forest Classifier performed better scores for this analysis, followed by
a close second K Nearest Neighbors. As a third place Decision Trees and last one Naive Bayes.

There is a lot of improvement work that can be done to this analysis:
    1) Run all this code several times and store the performance values and compute an average
        This to get a better idea of which classifier leads to better performance
    2) Run this code changing (adding/removing) variables to see find the best variables that help
       to predict the price of the Car
    3) Get in contact with the end customer of this analysis to get a better idea of the end use for
       and therefore which performance parameter would be the better representation of a good performance.
       (Example: Are False Positives more costly that False Negatives?)

In this analysis the probability threshold for the classifiers was taken as the default value 50%
since no more information was provided (input for client), but a good picture of how this might affect
the model was obtain from the ROC.

All in all even though more analysis could be done to improve the accuracy, at this point it seems
that using Random Forest or KNN would be a good approach for this problem on predicting if a Car would
be expensive or cheap based on the 10 attributes provided.

"""
#Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import *
import matplotlib.pyplot as plt

##############################################################################
############################ Cleaning data ###################################
##############################################################################

#Read in Data from the online archive
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

auto = pd.read_csv(url,header=None) ##This line was used to experiment with the local file 

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

#This section is to bin the Price into 2 Categories that in this case as an example will be named:
# 1 - Cheap, 2 - Expensive. This eventually will be binarized
NB = 2
bounds = np.linspace(np.min(auto["price"]), np.max(auto["price"]), NB+1)

def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= b[i-1])&(x < b[i])] = i
    
    y[x == b[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y

#This is a new attribute with the price variable binned into 2 categories
# 1 - Cheap 2 - Expensive (this is an oversimplification)    
auto["price_binned"] = bin(auto["price"], bounds) 

###############################################################################
######################## CLASSIFICATION MODELS ################################
###############################################################################

#Attributes
attributes_numeric = ["compression-ratio_norm","horsepower_norm","peak-rpm_norm","city-mpg_norm","highway-mpg_norm"]
attributes_categorical = ["Dual_Overhead","L_Engine","Overhead","Rotary","4wd","fwd","rwd","four","two","diesel","gas","hardtop","wagon","sedan","hatchback","convertible", "alfa-romero", "audi", "bmw", "chevrolet", "dodge", "honda", "isuzu", "jaguar", "mazda", "mercedes-benz", "mercury","mitsubishi", "nissan", "peugot", "plymouth", "porsche", "renault", "saab", "subaru", "toyota", "volkswagen", "volvo"]
attributes = attributes_numeric + attributes_categorical

#Split the data
variables = auto[attributes]
label = auto["price_binned"]
X_train, X_test, y_train, y_test = train_test_split(variables, label, test_size=0.30, random_state=42)

###############################################################################
################## Functions to Calculate Performance #########################
###############################################################################

#Function to calculate performance parameters based on Predictions and Actual values from the testing group
def perfomance_values(T,Y,header):
    print("\n\n#############\Classifier: #############\n", header)
    # Confusion Matrix
    CM = confusion_matrix(T, Y)
    print ("\n\nConfusion matrix:\n", CM)
    tn, fp, fn, tp = CM.ravel()
    print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
    AR = accuracy_score(T, Y)
    ER = 1.0 - AR
    P = precision_score(T, Y)
    R = recall_score(T, Y)
    F1 = f1_score(T, Y)
    return AR, ER, P, R, F1
    
###############################################################################

#Function to calculate the Receiver Operating Characteristic Curve
def ROM_plots(T,y):
    # ROC analysis
    LW = 2.0 # line width for plots
    LL = "lower right" # legend location
    LC = 'orange' # Line Color
    fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
    AUC = auc(fpr, tpr)
    print ("\nTP rates:", np.round(tpr, 2))
    print ("\nFP rates:", np.round(fpr, 2))
    print ("\nProbability thresholds:", np.round(th, 2))
    plt.figure()
    plt.title('Receiver Operating Characteristic curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPT')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
    plt.legend(loc=LL)
    plt.show()
    print ("\nAUC score (using auc function):", np.round(AUC, 2))
    print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")
    
#Function to make sure the output is a boolean as it is needed for the roc_curve
def booleans(vector):
    final = []
    for i in vector:
        if i == 1:
            i = 0
        else:
            i = 1
        final.append(i)
    return final

###############################################################################
#Function to train the model and obtain the perfomance values from each model (in a plot form)
def classifier_performance(V, header, X_train, X_test, y_train, y_test):
    accuracy_rate = []
    error_rate = []
    precision = []
    recall = []
    f1_score = []
    for i in range(len(header)):
        V[i].fit(X_train, y_train)
        proba = V[i].predict_proba(X_test)[:,1]#Predictions and probabilities
        prediction = V[i].predict(X_test)
        #Creating a Data Frame for the Test, Prediction and Probaility Data
        T = y_test.tolist()
        Y = prediction.tolist()
        y = proba.tolist()
        Y = booleans(Y)
        T = booleans(T)
        AR, ER, P, R, F1 = perfomance_values(T,Y,header[i])
        accuracy_rate.append(AR)
        error_rate.append(ER)
        precision.append(P)
        recall.append(R)
        f1_score.append(F1)
        ROM_plots(T,y)
    
    names = ['Accuracy Rate','Error Rate','Precision','Recall','F1 Score']
    list_of_values = [accuracy_rate, error_rate, precision, recall, f1_score]
    colors = ['indigo', 'tomato', 'dodgerblue', 'crimson', 'darkcyan']
    
    #Plots of each of the Performance Values for each of the Classifiers in this analysis
    for i in range(len(names)):
        plt.bar(header, list_of_values[i], color=colors[i])
        plt.xticks(rotation=45)
        plt.title(names[i])
        plt.ylabel("Score")
        plt.show()

###############################################################################
#################### Training & Applying the Model ############################
###############################################################################
#Parameters for Classifiers
k = 5 # number of neighbors
distance_metric = 'euclidean'
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter

#Classifiers
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric) #K Nearest Neighbors
nbc = GaussianNB() #Naive Bayes
dtc = DecisionTreeClassifier() #DecisionTree
rfc = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) #Random Forest

#List for Classifiers and Names
header = ["K_Nearest_Neighbors","Naive_Bayes","Decision_Tree","Random_Forest"]
V = [knn, nbc, dtc, rfc]

#Running and obtaining performance values from the Models listed above
classifier_performance(V, header, X_train, X_test, y_train, y_test)