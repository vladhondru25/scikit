from sklearn import datasets, model_selection
import numpy as np

iris = datasets.load_iris()

# Split in features and labels
X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

# Split training/testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)