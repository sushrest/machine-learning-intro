# scikit-learn Machine Learning Library in python 
# Environment Python Tensorflow

# Following example demonstrates a basic Machine Learning examples using 
# Supervised Learning by making use of Decision Tree Classifier and its fit algorithm to predict whether the given
# features belong to Apple or Orange
from sklearn import tree

# Preparing data for the decision tree classifier
# features as an input for classifier
features = [
    [140, 1], #140 as weight in grams and 1 as Bumpy surface and 0 as Smooth surface
    [130, 1], 
    [150, 0], 
    [170, 0]
    ]

# labels as an output for classifier 
labels = [0, 0, 1, 1] # 0 as an apple and 1 as orange 

print 'Marking features type to Int by'
print '1 as Smooth and 0 as Bumpy and 0 as Apple and 1 as Orange'
print ' '

# Initializing a classifier this can be treated as an empty box of Rule.
clf = tree.DecisionTreeClassifier()

print '.'
print '.'

print 'Learning algorithm is just a procedure that creates classifier such as DecisionTree.'

print '.'
print '.'
print ' '
print 'Calling a built-in algorithm called fit from DecisionTreeClassifier Object '
print '.'
print ' '
print 'Think of fit being a synonym for Fine Pattern and Data'
print '.'
print ' '
clf = clf.fit(features, labels)

print 'Now lets Predict'
print '1'
print '2.. and '
print 'BAAAM !!'

print 'Predicting a fruit which is 160g and Bumpy = 0 '
print clf.predict([[160,0]])
print 'If 0 its Apple and if 1 its Orange'