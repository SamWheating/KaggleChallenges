import pandas as pd
import numpy as np
import math
import time
import csv

MODE = "test"
AVG_KPS = 0.00400644693588925

"""

First ever submission to a Kaggle contest.
Compute the average rate of travel based on hypersine (as-the-crow-flies) distance
Calculate predictions based on (distance / average speed). 

This is not a good model. Test Data RMSLE = 0.70380. 

"""

# haversine: returns the haversine(spherical) distance between two points.  

def haversine(startLong, startLat, endLong, endLat):
	lat1 = math.radians(startLat)
	lat2 = math.radians(endLat)
	long1 = math.radians(startLong)
	long2 = math.radians(endLong)
	dLat = lat2 - lat1
	dLong = long2 - long1
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1) * math.cos(lat2) * math.sin(dLong/2) * math.sin(dLong/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = 6371 * c   # radius of earth
	return d

start = time.time()

# Load Training Data from .csv

if MODE == "test":
	data = pd.read_csv('test.csv')

	num_entries = data.shape[0]

	output = [0 for i in range(num_entries+1)]
	output[0] = ["id", "trip_duration"]

	i = 1

	for index, row in data.iterrows():
		distance = haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude'])
		trip_id = row['id']
		duration_prediction = math.floor(distance / AVG_KPS)
		output[i] = [trip_id, duration_prediction]		
		i += 1

	my_df = pd.DataFrame(output)

	my_df.to_csv('output.csv', index=False, header=False)

if MODE == "train":
	data = pd.read_csv('train.csv')

	sum = 0

	for index, row in data.iterrows():
		distance = haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude'])
		duration = row['trip_duration']
		kps = distance / duration
	#	print("distance = ", str(distance), " duration = ", str(duration), " km / second = ", str(kps))
		sum += kps

	print("average kps = ", str(sum/num_entries))

	end = time.time()

	print("calc duration = ", str(end-start))
