from PIL import Image
import mnist
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


# Get dataset
x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()
# Reshape the data from 2D images to 1D arrays
x_train = x_train.reshape((-1,28*28))
x_test = x_test.reshape((-1,28*28))
# Rescale the data to be uniform
x_train = x_train / 256
x_test = x_test / 256

# Define the model (optimiser, neural network architecture)
clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64))
# Training
clf.fit(x_train, y_train)

# Evaluating
predictions = clf.predict(x_test)
# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, predictions)
print(conf_mat)
# Compute accuracy
acc = conf_mat.trace() / conf_mat.sum()
print("Accuracy = {:.2f}%".format(100*acc))
