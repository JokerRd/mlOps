import pandas as pd

data = pd.read_csv('titanic.csv')
new_data = data.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
new_data.to_csv('titanic.csv', index=False)