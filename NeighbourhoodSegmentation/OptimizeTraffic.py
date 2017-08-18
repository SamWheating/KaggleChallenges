import pandas as pd
import numpy as np
import math
import time
import csv
import datetime

"""

Machine Learning / Optimization;
Test a Variety of Values for Traffic / no traffic time multiplier. 
Return Best Values. 

"""

data = pd.read_csv('trainlabelled.csv')
key = pd.read_csv('key.csv')
num_entries = data.shape[0]

traffic = np.arange(1.04, 1.15, 0.01)
noTraffic = np.arange(0.9, 1.02, 0.01)

NUM_SAMPLES = 5000

def isTraffic(tstamp):

	tstamp2 =  datetime.datetime.strptime(tstamp, "%Y-%m-%d %H:%M:%S")

	if int(datetime.datetime.strftime(tstamp2, "%w"))	in [0,6]:
		return False
	if int(datetime.datetime.strftime(tstamp2, "%H"))	in range(6, 18):
		return True
	return False


def logLoss(predictions, labels):
	sum = 0
	for i in range(len(predictions)):
		sum = sum + (math.log1p(predictions[i]) - math.log1p(labels[i]))**2
	sum = sum / len(predictions)
	return (math.sqrt(sum))

# Run test: performs validation test on 5000 samples and returns log loss.

def runTest(trafficConstant, noTrafficConstant):
	validationData = data.sample(NUM_SAMPLES)
	predictions = [0 for i in range(NUM_SAMPLES)]
	labels = [0 for i in range(NUM_SAMPLES)]
	i = 0
	for index, row in validationData.iterrows():
		from_hood = row['pickup']
		kps = float(key.iloc[from_hood]["from_kps"])
		distance = row['distance']
		duration_prediction = (distance / kps)

		if isTraffic(row['time']):
			duration_prediction = math.floor(duration_prediction * trafficConstant)
		else:
			duration_prediction = math.floor(duration_prediction * noTrafficConstant)

		predictions[i] = duration_prediction
		labels[i] = row['trip_duration']

		i += 1

	return logLoss(predictions, labels)


start = time.time()

# TRAINING:
# Load Labelled set from CSV. For each From / To Destination add the KPS to an array. 

results = [10,0,0]

for a in traffic:
	for b in noTraffic:
		result = runTest(a, b)
		print("with traffic = ", str(a), " and noTraffic ", str(b), "LogLoss = ", str(result))
		if result < results[0]:
			results[0] = result
			results[1] = a
			results[2] = b

end = time.time()

print("test concluded in ", str(end-start), " seconds.")
print("best result: ")
print("with traffic = ", str(results[1]), " and noTraffic ", str(results[2]), "LogLoss = ", str(results[0]))