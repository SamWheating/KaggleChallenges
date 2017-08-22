import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

VALIDATE = True
TEST = not VALIDATE

# Importing data

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

# Cleaning Data

# Fill Holes:

trainData['Embarked'].fillna("C", inplace=True)
testData['Embarked'].fillna("C", inplace=True)
testData['Fare'].fillna(32.2, inplace=True)		# Fill missing fares with mean fare 32.2

# Re-encode ordinal variables as integers:

embarked = ['S', 'C', 'Q']
sex = ['male', 'female']

for i, row in trainData.iterrows():
	a = embarked.index(row['Embarked'])
	trainData.set_value(i, 'Embarked', a)
	a = float(sex.index(row['Sex']))
	trainData.set_value(i, 'Sex', a)

for i, row in testData.iterrows():
	a = embarked.index(row['Embarked'])
	testData.set_value(i, 'Embarked', a)
	a = float(sex.index(row['Sex']))
	testData.set_value(i, 'Sex', a)

# Split Data into training and Validation Sets, both split into x and y

train, validation = train_test_split(trainData, test_size = 0.2) 

train_x = train.iloc[:, [2,4,6,7,9,11]]
validation_x = validation.iloc[:, [2,4,6,7,9,11]]
train_y = list(train.iloc[:, 1])
validation_y = list(validation.iloc[:, 1])
test_x = testData.iloc[:, [1,3,5,6,8,10]]

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_x, train_y)

num_test = test_x.shape[0]
output = [0 for i in range(num_test)]

if VALIDATE:

	num_entries = len(validation_y)
	correct = 0
	j = 0
	for i, row in validation_x.iterrows():
		prediction = clf.predict(row.reshape(1,-1))
		if prediction == int(validation_y[j]): 
			correct += 1
		j += 1

	accuracy = (100 * correct / num_entries)
	print("Validation Accuracy: ", accuracy, "%")

if TEST:
	for i, row in test_x.iterrows():
		prediction = clf.predict(row.reshape(1,-1))
		entry = [i, int(prediction)]
		output[i] = entry	

	my_df = pd.DataFrame(output)
	my_df.to_csv('Output.csv', index=False, header=["PassengerId", "Survived"])
