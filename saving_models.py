import joblib
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5)

# Save model
filename = 'model.sav'
joblib.dump(clf, filename)

# Load model
clf = joblib.load(filename)