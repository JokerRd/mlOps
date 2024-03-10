from sklearn.linear_model import LogisticRegression
import pandas as pd
from joblib import dump

X_train = pd.read_csv("train/X_train.csv", header=0)
y_train = pd.read_csv("train/y_train.csv")

reg_model = LogisticRegression()
reg_model.fit(X_train, y_train)

dump(reg_model, "model.joblib")
