# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. Load Iris data
X, y = load_iris(return_X_y=True)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Train
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

# 4. Save as model.pkl
joblib.dump(clf, 'model.pkl')
print("Model saved to model.pkl")
