# Code basic pipeline for Supervised Learning

#import a dataset 
from sklearn import datasets 

# using dataset tool to load Iris flower data
iris = datasets.load_iris()

X = iris.data
y = iris.target 

# Partitioning the imported data to train and test 
from sklearn.cross_validation import train_test_split

# Splitting half using test_size = .5 so 50% to test and 50% to train
# Here X_train and y_train for train label and train target and vice versa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Using DecisionTree Classifier 
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# Using KNeighbors Classifier 
# disable DecisionTree Classifier to test this.
# Just by enabling and disabling the 2 lines of code by switching between Classifiers
# We can get similar results, this shows that switching between different 
# types of classifier 
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# Now testing and calculating accuracy 
# We can compare the true label with the predicted label 
# In sci-kit there is a convinient method called accuracy_score 

from sklearn.metrics import accuracy_score

print accuracy_score(y_test, predictions)

# What it means to learn from data 
# Features    Label
#   f(x)    =   y
# def classify(features): 
#   # do some logic
#   return label

# Equation of a line
# y=mx+b #where m is slope and b is y intercept
# 
# One way to think of training it to adjust the parameters of the model