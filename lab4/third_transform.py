import pandas as pd

data = pd.read_csv('titanic.csv')
one_hot_encoded_data = pd.get_dummies(data, columns=['Sex'])
one_hot_encoded_data.to_csv('titanic.csv', index=False)