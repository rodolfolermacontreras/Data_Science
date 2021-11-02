# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:54:25 2020

@author: Rodolfo Lerma

Specifically, you’ll need to pick two (or more) classifiers and perform each one of the following tasks, in order:

    Split your dataset into training and testing sets
    Train your classifiers, using the training set partition
    Apply your (trained) classifiers on the test set
    Measure each classifier’s performance using at least 3 of the metrics we covered in this course 
   (one of them has to be the ROC-based one). At one point, you’ll need to create a confusion matrix.
    Document your results and your conclusions, along with any relevant comments about your work

Make sure that you state any assumptions you make (e.g. the parameters you use for the classifiers) in your report. 
Also, for this assignment you’ll need to provide your Python code, along with comments to show what each part of the script does. 
For additional traction with this material, you can examine how the performance of the winning classifier changes if you 
bin one of the variables you’ve used (you’ll need to binarize the binned variable after you create it).

Note: if you have a multi-class problem, you can focus on a particular class, when evaluating your classifier. 
This way, you can simplify the problem and turn it into a binary one 
(class 1 = class you are focusing on, class 0 = any other class). 
This will enable you to use a confusion matrix for your evaluation of the classifier.

"""
#Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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
# 1 - Cheap, 2 - Expensive
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

attributes_numeric = ["compression-ratio_norm","horsepower_norm","peak-rpm_norm","city-mpg_norm","highway-mpg_norm"]
attributes_categorical = ["Dual_Overhead","L_Engine","Overhead","Rotary","4wd","fwd","rwd","four","two","diesel","gas","hardtop","wagon","sedan","hatchback","convertible", "alfa-romero", "audi", "bmw", "chevrolet", "dodge", "honda", "isuzu", "jaguar", "mazda", "mercedes-benz", "mercury","mitsubishi", "nissan", "peugot", "plymouth", "porsche", "renault", "saab", "subaru", "toyota", "volkswagen", "volvo"]
attributes = attributes_numeric + attributes_categorical

#Split the data
variables = auto[attributes]
label = auto["price_binned"]
X_train, X_test, y_train, y_test = train_test_split(variables, label, test_size=0.30, random_state=42)

################## Functions to Calculate Performance ########################
###############################################################################
def perfomance_values(T,Y,header):
    print("\n\n#############\Classifier: #############\n", header)
    # Confusion Matrix
    CM = confusion_matrix(T, Y)
    print ("\n\nConfusion matrix:\n", CM)
    tn, fp, fn, tp = CM.ravel()
    print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
    AR = accuracy_score(T, Y)
    print ("\nAccuracy rate:", AR)
    ER = 1.0 - AR
    print ("\nError rate:", ER)
    P = precision_score(T, Y)
    print ("\nPrecision:", np.round(P, 2))
    R = recall_score(T, Y)
    print ("\nRecall:", np.round(R, 2))
    F1 = f1_score(T, Y)
    print ("\nF1 score:", np.round(F1, 2))
###############################################################################
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
#################### Training & Applying the Model ############################
###############################################################################
header = "K Nearest Neighbors classifier"
#print ('\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X_train, y_train.astype('int'))
#Predictions and probabilities
proba = knn.predict_proba(X_test)[:,1]
prediction = knn.predict(X_test)
#print ("\nPredictions for test set:\n")
#print (prediction)
#print ('\nActual class values:\n')
#print (y_test.tolist())
#Creating a Data Frame for the Test, Prediction and Probaility Data
T = y_test.tolist()
Y = prediction.tolist()
y = proba.tolist()
Y = booleans(Y)
T = booleans(T)
perfomance_values(T,Y,header)
ROM_plots(T,y)
###############################################################################
header =  "Naive Bayes classifier"
#print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X_train, y_train)
proba1 = nbc.predict_proba(X_test)[:,1]
prediction1 = nbc.predict(X_test)
#print ("predictions for test set:")
#print (prediction1)
#print ('actual class values:')
#print (y_test)
R = prediction1.tolist()
R = booleans(R)
r = proba1.tolist()
perfomance_values(T,R,header)
ROM_plots(T,r)
###############################################################################
'''header = "Support vector machine classifier"
t = 0.001 # tolerance parameter
kp = 'rbf' # kernel parameter
#print ('\n\nSupport Vector Machine classifier\n')
clf = SVC(kernel=kp, tol=t)
clf.fit(X_train, y_train)
proba2 = clf.predict_proba(X_test)[:,1]
prediction2 = clf.predict(X_test)
#print ("predictions for test set:")
#print (clf.predict(X_test))
#print ('actual class values:')
#print (y_test)
Q = prediction2.tolist()
Q = booleans(Q)
q = proba2.tolist()
perfomance_values(T,Q,header)
ROM_plots(T,q)'''
###############################################################################
header =  "Decision Tree classifier"
#print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X_train, y_train)
proba3 = clf.predict_proba(X_test)[:,1]
prediction3 = clf.predict(X_test)
#print ("predictions for test set:")
#print (clf.predict(X_test))
#print ('actual class values:')
#print (y_test)
U = prediction3.tolist()
U = booleans(U)
u = proba3.tolist()
perfomance_values(T,U,header)
ROM_plots(T,u)
###############################################################################
header = "Random Forest classifier"
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
#print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X_train, y_train)
proba4 = clf.predict_proba(X_test)[:,1]
prediction4 = clf.predict(X_test)
#print ("predictions for test set:")
#print (clf.predict(X_test))
#print ('actual class values:')
#print (y_test)
W = prediction4.tolist()
W = booleans(W)
w = proba4.tolist()
perfomance_values(T,W,header)
ROM_plots(T,w)