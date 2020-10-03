from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Get Boston dataset
boston = datasets.load_boston()

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Visualise first feature 
plt.scatter(X[:,0],y)
plt.show()


# Train
l_reg = linear_model.LinearRegression()
model = l_reg.fit(X_train, y_train)
# Evaluate
predictions = model.predict(X_test)
print("Coefficient of determination R^2 value: ", l_reg.score(X, y))
print("Coefficient: ", l_reg.coef_)
print("Intercept: ", l_reg.intercept_)
