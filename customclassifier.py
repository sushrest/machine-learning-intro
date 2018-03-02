# This script has a custom written classifier

# First we will import sklearn datasets
# Then we will split the data into test and train 
# We will use training data to test the classifier
# And use the test data to test the classifier to see how accurate it was
# 

import random

# Implementing custom classifier which is based on k-NN (k nearsneighbors) nearest neighbours
# classifier where K is the number of neighbour when we consider for example if K is 3 then
# we take 3 of the closest point to the test data and the label that satisfies the majority (which is 2)
# of the training data will be predicted as the label for the test data.
# First we need to find the nearest neighbour

# We use the straight line distance formula called Euclidean Distance 
# Two dimension space 
# d(p,q) = SquareRoot (q1-p1)2 + (q2-p2)2
# 
# Three dimension space 
# d(p,q) = SquareRoot (q1-p1)2 + (q2-p2)2 + (qn-pn)2 
#
# As the feature increases we just add more terms to the equation 4, 5, 6 dimensions and so on

# Import a library called scipy
from scipy.spatial import distance

# Define a distance method 
def euc(a,b): # here b is a point from test data and a is from training data
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
        
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)