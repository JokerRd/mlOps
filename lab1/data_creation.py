from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import os

iris = fetch_ucirepo(id=53)

iris_data = iris.data.features
iris_target = iris.data.targets

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target)

os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

X_train.to_csv('train/X_train.csv', index=False)
X_test.to_csv('test/X_test.csv', index=False)
y_train.to_csv('train/y_train.csv', index=False)
y_test.to_csv('test/y_test.csv', index=False)
