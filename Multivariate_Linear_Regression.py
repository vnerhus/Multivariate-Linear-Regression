## ------ IMPORTS ------ ##
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


## ------ CREDITING ------ ##
print("\n## Multivariate Linear Regression | by Vegard Nerhus ##")
print("-------------------------------------------------------")


## ------ INITIALIZATION ------ ##
# Hyperparameters
alpha = 0.003
n_iterations = 100000

# Load data X and Y, a labeled trainingset
diabetes_data = load_diabetes()
x_raw = np.array(diabetes_data['data'])
y = np.array(diabetes_data['target'])

#Split trainingset into training and testsets
x_train, x_test, y_train, y_test = train_test_split(x_raw, y, test_size=0.3)
m_train = len(y_train)
m_test = len(y_test)

# Add bias column, a column of ones, to x_train and x_test
def addBiasColumn(x,m):
    xnew = np.ones((m, len(x[0])+1))
    xnew[:, 1:len(xnew[0])] = x
    return xnew

x_train = addBiasColumn(x_train, m_train)
x_test = addBiasColumn(x_test, m_test)

# Initialize theta
np.random.seed(1)
theta = np.random.rand(len(x_train[0]))

# Compute and display initial cost, J(theta)
def computeCost(x, y, theta, m):
    J = np.dot(np.transpose(np.dot(x,theta) - y), (np.dot(x,theta) - y))
    return J / (2*m)

print("INITIAL COST: %f" % computeCost(x_train, y_train, theta, m_train))


## ------ TRAINING OF MODEL ------ ##
# Run gradient descent
for i in range(n_iterations):
    h = np.dot(x_train, theta) # Hypothesis
    error = h-y_train # Hypothesis (prediction) - target
    theta = theta - (alpha/m_train) * np.dot(x_train.T, error)
    if(i%10000 == 0):
        print("Iteration: %i | Cost: %f" % (i, computeCost(x_train, y_train, theta, m_train)))


## ------ FINAL RESULTS ------ ##
print("Iteration: %i | Cost: %f" % (n_iterations, computeCost(x_train, y_train, theta, m_train)))
print("\nCOST ON TESTSET: %f\n" % computeCost(x_test, y_test, theta, m_test))
