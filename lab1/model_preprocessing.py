from sklearn.preprocessing import StandardScaler
import pandas as pd

X_train = pd.read_csv("train/X_train.csv")
X_test = pd.read_csv("test/X_test.csv")

X_scaled_train = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_scaled_test = pd.DataFrame(StandardScaler().fit_transform(X_test), columns=X_test.columns, index=X_test.index)

X_scaled_train.to_csv("train/X_train.csv", index=False)
X_scaled_test.to_csv("test/X_test.csv", index=False)