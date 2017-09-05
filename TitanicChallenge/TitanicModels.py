import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm
import numpy as np


# CONTROLS
# Change these variables to control the test and validation process.

VALIDATE = False				# True for Validation, False for Testing. 
NUM_TESTS = 2					# of of iterations of validation process.
ALG = "Gaussian"				# Can be KNN, RandomForest, MLP, SVC, Gaussian

KNN_PARAMS = dict(n_neighbors=5)
RANDOMFOREST_PARAMS = dict(max_depth=7, random_state=0, bootstrap=True, criterion='gini')
MLP_PARAMS = dict(solver='lbfgs', alpha=5e-7, hidden_layer_sizes=(4,1), random_state=1, activation='logistic')
SVC_PARAMS = dict(penalty='l2', dual=False)
GAUSSIAN_PARAMS = dict(kernel=1.0 * RBF(length_scale=1.0))

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


num_test = testData.shape[0]
output = [0 for i in range(num_test)]

if VALIDATE:

	sum_accuracy = 0.0

	for test_number in range(NUM_TESTS):

		train, validation = train_test_split(trainData, test_size = 0.1) 

		train_x = train.iloc[:, [2,4,5,9]]					# only keep Sex, Age, Fare and Pclass
		validation_x = validation.iloc[:, [2,4,5,9]]
		train_y = list(train.iloc[:, 1])
		validation_y = list(validation.iloc[:, 1])

		if ALG == "KNN":
			clf = KNeighborsClassifier(**KNN_PARAMS)
			clf.fit(train_x, train_y)

		if ALG == "RandomForest":
			clf = RandomForestClassifier(**RANDOMFOREST_PARAMS)
			clf.fit(train_x, train_y)

		if ALG == "MLP":
			clf = MLPClassifier(**MLP_PARAMS)
			clf.fit(train_x, train_y)

		if ALG == "SVC":
			clf = svm.LinearSVC(**SVC_PARAMS)
			clf.fit(train_x, train_y)

		if ALG == "Gaussian":
			clf = GaussianProcessClassifier(**GAUSSIAN_PARAMS)
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

	print("Total accuracy over ", str(NUM_TESTS), " tests: ", str(sum_accuracy/NUM_TESTS))

if not VALIDATE:

	train_x = trainData.iloc[:, [2,4,5,9]]			# only keep Sex, Age, Fare and Pclass
	train_y = list(trainData.iloc[:, 1])
	test_x = testData.iloc[:, [1,3,4,8]]

	if ALG == "KNN":
		clf = KNeighborsClassifier(**KNN_PARAMS)
		clf.fit(train_x, train_y)

	if ALG == "RandomForest":
		clf = RandomForestClassifier(**RANDOMFOREST_PARAMS)
		clf.fit(train_x, train_y)

	if ALG == "MLP":
		clf = MLPClassifier(**MLP_PARAMS)
		clf.fit(train_x, train_y)

	if ALG == "SVC":
		clf = svm.LinearSVC(**SVC_PARAMS)
		clf.fit(train_x, train_y)

	if ALG == "Gaussian":
		clf = GaussianProcessClassifier(**GAUSSIAN_PARAMS)
		clf.fit(train_x, train_y)

	for i, row in test_x.iterrows():
		prediction = clf.predict(row.values.reshape(1,-1))
		entry = [i+892, int(prediction)]
		output[i] = entry	

	my_df = pd.DataFrame(output)
	my_df.to_csv('Output.csv', index=False, header=["PassengerId", "Survived"])
