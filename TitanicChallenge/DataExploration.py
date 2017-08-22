import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

"""
Used as a sandbox for testing data cleaning and feature ranking. 

"""


# Importing data

trainData = pd.read_csv('train.csv')

trainData['Age'].fillna(trainData['Age'].mean(), inplace=True)
trainData['Fare'].fillna(trainData['Fare'].mean(), inplace=True)
trainData['Embarked'].fillna('S', inplace=True)

trainData.drop('Name', axis=1)
trainData.drop('Cabin', axis=1)
trainData.drop('Ticket', axis=1)

embarked = ['S', 'C', 'Q']
sex = ['male', 'female']

for i, row in trainData.iterrows():
	a = embarked.index(row['Embarked'])
	trainData.set_value(i, 'Embarked', a)
	a = float(sex.index(row['Sex']))
	trainData.set_value(i, 'Sex', a)

print(trainData)

train_x = trainData.iloc[:, [2,4,5,6,7,9,11]]
train_y = list(trainData.iloc[:, 1])

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(train_x, train_y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
featureNames = list(train_x.columns.values)

for f in range(train_x.shape[1]):
    print("%d. %s (%f)" % (f + 1, featureNames[indices[f]], importances[indices[f]]))

"""
Feature Importance:
1. Sex (0.289760)
2. Age (0.245238)
3. Fare (0.236738)
4. Pclass (0.107579)
5. SibSp (0.044878)
6. Parch (0.043768)
7. Embarked (0.032039)
"""