import pandas as pd
import numpy as np
import math
import time
import csv
import datetime

"""

Lookup averaged kps for neighbourhood of trip, apply to trip distance. 
Run Algorithm on test set, including fixed multiplier based on workday traffic or not. 

"""

RUSH_HOUR_CONSTANT = 1.08
WEEKEND_CONSTANT = 0.99

def isTraffic(tstamp):

	tstamp2 =  datetime.datetime.strptime(tstamp, "%Y-%m-%d %H:%M:%S")

	if int(datetime.datetime.strftime(tstamp2, "%w"))	in [0,6]:
		return False
	if int(datetime.datetime.strftime(tstamp2, "%H"))	in range(6, 18):
		return True
	return False

start = time.time()

# TRAINING:
# Load Labelled set from CSV. For each From / To Destination add the KPS to an array. 

data = pd.read_csv('testlabelled.csv')
key = pd.read_csv('key.csv')

num_entries = data.shape[0]

output = [0 for i in range(num_entries)]

i = 0

for index, row in data.iterrows():
	trip_id = row['id']
	from_hood = row['pickup']
	kps = float(key.iloc[from_hood]["from_kps"])
	distance = row['distance']
	duration_prediction = (distance / kps)

	if isTraffic(row['time']):
		duration_prediction = math.floor(duration_prediction * RUSH_HOUR_CONSTANT)
	else:
		duration_prediction = math.floor(duration_prediction * WEEKEND_CONSTANT)

	output[i] = [trip_id, duration_prediction]		
	i += 1

my_df = pd.DataFrame(output)

my_df.to_csv('neighbourhoodOutput4.csv', index=False, header=["id", "trip_duration"])

