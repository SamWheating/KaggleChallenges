import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

VALIDATE = False
ALG = "RandomForest"			# Can be KNN, RandomForest

# Importing data

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

# Cleaning Data

# Fill Holes:
trainData['Age'].fillna(trainData['Age'].mean(), inplace=True)
testData['Age'].fillna(testData['Age'].mean(), inplace=True)
trainData['Embarked'].fillna("S", inplace=True)
testData['Embarked'].fillna("S", inplace=True)
testData['Fare'].fillna(32.2, inplace=True)		# Fill missing fares with mean fare 32.2


# Re-encode ordinal variables as integers:

embarked = ['S', 'C', 'Q']
sex = ['male', 'female']

for i, row in trainData.iterrows():
	a = float(sex.index(row['Sex']))
	trainData.set_value(i, 'Sex', a)

for i, row in testData.iterrows():
	a = float(sex.index(row['Sex']))
	testData.set_value(i, 'Sex', a)

# Split Data into training and Validation Sets, both split into x and y

train, validation = train_test_split(trainData, test_size = 0.2) 

num_test = testData.shape[0]
output = [0 for i in range(num_test)]

if VALIDATE:

	NUM_TESTS = 50

	sum_accuracy = 0.0

	for test_number in range(NUM_TESTS):

		train, validation = train_test_split(trainData, test_size = 0.1) 

		train_x = train.iloc[:, [2,4,5,9]]
		validation_x = validation.iloc[:, [2,4,5,9]]
		train_y = list(train.iloc[:, 1])
		validation_y = list(validation.iloc[:, 1])

		if ALG == "KNN":
			clf = KNeighborsClassifier(n_neighbors=5)
			clf.fit(train_x, train_y)

		if ALG == "RandomForest":
			clf = RandomForestClassifier(max_depth=2, random_state=0)
			clf.fit(train_x, train_y)

		num_entries = len(validation_y)
		correct = 0
		j = 0
		for i, row in validation_x.iterrows():
			prediction = clf.predict(row.values.reshape(1,-1))
			if prediction == int(validation_y[j]): 
				correct += 1
			j += 1

		accuracy = (100 * correct / num_entries)
		sum_accuracy += accuracy
#			print("Test #", test_number, " Validation Accuracy: ", accuracy, "%")

	print("Total accuracy over 10 tests: ", str(sum_accuracy/NUM_TESTS))

if not VALIDATE:

	train_x = trainData.iloc[:, [2,4,5,9]]
	train_y = list(trainData.iloc[:, 1])
	test_x = testData.iloc[:, [1,3,4,8]]

	if ALG == "KNN":
		clf = KNeighborsClassifier(n_neighbors=9)
		clf.fit(train_x, train_y)

	if ALG == "RandomForest":
		clf = RandomForestClassifier(max_depth=5, random_state=0)
		clf.fit(train_x, train_y)

	for i, row in test_x.iterrows():
		prediction = clf.predict(row.values.reshape(1,-1))
		entry = [i+892, int(prediction)]
		output[i] = entry	

	my_df = pd.DataFrame(output)
	my_df.to_csv('Output.csv', index=False, header=["PassengerId", "Survived"])



"""
TO DO:

Automate validation process to repeat 10x for higher validation accuracy.
Include additional features. 
Implement PCA?
Try KNN?

"""