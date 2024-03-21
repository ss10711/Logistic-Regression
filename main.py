# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""""
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
"""
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from numpy import log, dot, e
from numpy.random import rand


# Loading data
def convertData():
    df = load_breast_cancer(as_frame=True)
    data = pd.DataFrame(data=df.data)
    data["target"] = df.target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test


# sigmoid/activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# b= bias, w=weight
# z = x*w +b

# h_theta = sigmoid(z)
# h(x) = line equation
def param(x_train):
    features = x_train.shape[1]
    w = np.zeros(features + 1)
    b = 0
    return w, b





def cost_functionIndividual(y_true, y_hat):
    # Compute individual losses for each sample
    individual_losses = - (y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

    return individual_losses

def MinMaxNorm(X):
    X.reset_index(drop=True, inplace=True)
    # Assuming 'X' is your DataFrame with shape (426, 31)
    # Calculate the minimum and maximum values for each column
    X_min = X.min()
    X_max = X.max()

    # Apply Min-Max scaling
    normalized_X = (X - X_min) / (X_max - X_min)

    return normalized_X



# Finds gradient indirectly by computing the element wise multiplication of the
# input features (x) with the difference between the predicted labels (y_hat) and the labels(label).
# Scales the result by the learning rate divided by the number of samples




def fitDerivativeSigmoid(X, y, epochs=100, lr=0.1):
    weights, bias = param(X)
    X = MinMaxNorm(X)
    # Add column of zeros for bias
    ones = np.ones((len(X),1))
    X.insert(loc=0, column='ones',  value=ones)

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    N = len(X)
    for _ in range(epochs):
        # Compute sigmoid function for epochs
        sum_error = 0
        # y_hat_epoch = sigmoid(dot(X, weights.T))
        for i in range(len(X)):

            row = X.iloc[i]
            label = y[i]
            # Compute linear combination zi = w * xTi
            zi = dot(row, weights.T)
            # Apply sigmoid function to compute predicted probability
            y_hat = sigmoid(zi)  # returns 1 number between 0

            sum_error += cost_functionIndividual(label, y_hat)

            # Updates weights using gradient descent
            for c in range(len(weights)):
                weights[c] = weights[c] - lr * (y_hat - label) * row[c]
            # loss = cost_functionIndividual(label,y_hat)

        # Saving Progress
        loss = sum_error/N
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (_, lr, loss))

    return weights


def predict(self, X):
    # Predicting with sigmoid function
    z = dot(X, self.weights)
    # Returning binary result
    return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]


x_train, x_test, y_train, y_test = convertData()

weights = fitDerivativeSigmoid(x_train, y_train)

print("Final Weights:", weights)
