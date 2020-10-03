import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Read data
data = pd.read_csv('car.data')
# print(data.head())

X = data[['buying','maint','safety']].values
y = data[['class']]
# print(X,y)


# Converting data - encoding the labels
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = Le.fit_transform(X[:,i])

# print(X)

# Converting data - label mapping
label_mapping = {
    'unacc': 0,
    'acc'  : 1,
    'good' : 2,
    'vgood': 3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)

# print(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

accuracies = []

starting_k = 1
ending_k   = 75

# Test for multiple values of k
for k in range(starting_k,ending_k):
    # Create model
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    # Train model
    knn.fit(X_train,y_train[:,0])
    # Evaluate model
    prediction = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    # print("Predictions: ",  prediction)
    # print("Accuracy: ", accuracy)

    accuracies.append(accuracy)


plt.plot(list(range(starting_k,ending_k)),accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.ylim(top=0.8,bottom=0.6)
plt.show()