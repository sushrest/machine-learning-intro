# Following piece of code generates a histogram diagram for two type of dogs 
# 500 Labs and 500 Greyhounds using their randomly generated heights using numpy
import numpy as np
import matplotlib.pyplot as plt 

greyhounds = 500
labs = 500

# creating random 500 data for greyhound with average height of 28
grey_height = 28 + 4 * np.random.randn(greyhounds)

# creating random 500 data for greyhound with average height of 24
lab_height = 24 + 4 * np.random.randn(labs)

# Generate a histogram diagram with greyhound on red and labrador on blue
plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()

# In the diagram the probability of a dog to be a Lab or a Greyhound can be distinguished if the feature 
# are not close enough but if the feature are close the probability is not perfect. Thats why in Machine Learning
# we always need multiple features to contribute to the probability accuracy or prediction accuracy.
