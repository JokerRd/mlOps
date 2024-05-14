import pandas as pd

data = pd.read_csv('titanic.csv')
data['Age'].fillna(data['Age'].mean(), inplace=True)
data.to_csv('titanic.csv', index=False)