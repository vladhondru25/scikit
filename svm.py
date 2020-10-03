from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()

# Split in features and labels
X = iris.data
y = iris.target

# Split training/testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']


# Create model
model = svm.SVC()
# Train
model.fit(X_train, y_train)
# Evaluate
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("Accuracy = {:.2f}%".format(acc*100))