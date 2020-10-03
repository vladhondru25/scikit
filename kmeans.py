from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd


# Load dataset
bc = load_breast_cancer()

x = scale(bc.data)
y = bc.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Train
model = KMeans(n_clusters=2, random_state=0)
model.fit(x_train)

# Evaluate
predictions = model.predict(x_test)
print('Accuracy = {:.2f}%'.format( 100*accuracy_score(y_test,predictions) ))