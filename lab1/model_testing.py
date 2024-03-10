from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score

reg_model = load("model.joblib")

X_test = pd.read_csv("test/X_test.csv", header=0)
y_test = pd.read_csv("test/y_test.csv", header=0)

y_pred = reg_model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"Model test accuracy is: {score:.3f}")

