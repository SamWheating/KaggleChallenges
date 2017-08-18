import pandas as pd
import numpy as np
import math
import time
import csv

"""
Divide New York into a grid based on longitude and latitude. 
Calculate average speed for every combination of pickup / dropoff points.
Re-label test and training Data with to / from neighbourhoods. 

"""

start = time.time()

NUM_SEGMENTS = 10

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

data = pd.read_csv('test.csv')

num_entries = data.shape[0]

# segment latitude into groups

sortedPickupLat = data.sort_values("pickup_latitude")
interval = math.floor(num_entries / NUM_SEGMENTS)
i = 1
lat_segments = []
while(i < NUM_SEGMENTS):
	segment = sortedPickupLat.iloc[(i*interval)]["pickup_latitude"]
	lat_segments += [segment]
	i += 1

# segment longitude into groups

sortedPickupLong = data.sort_values("pickup_longitude")
i = 1
long_segments = []
while(i < NUM_SEGMENTS):
	segment = sortedPickupLong.iloc[(i*interval)]["pickup_longitude"]
	long_segments += [segment]
	i += 1

# bin Data into 'neighbourhoods'

def assignSector(number, divisors):
	if number < divisors[0]: return 0
	i = 1
	while(i < NUM_SEGMENTS-1):
		if(number < divisors[i]):
			return i
		i += 1
	return NUM_SEGMENTS-1


tempdata = [[0 for i in range(5)] for i in range(num_entries)]

i = 0
for index, row in data.iterrows():
	pickup = str(assignSector(row["pickup_longitude"], long_segments)) + str(assignSector(row["pickup_latitude"], lat_segments))
	dropoff = str(assignSector(row["dropoff_longitude"], long_segments)) + str(assignSector(row["dropoff_latitude"], lat_segments))
	distance = haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude'])
	tempdata[i] = [row['id'], row['pickup_datetime'], distance, pickup, dropoff]
	i += 1
	if(i % 100 == 0): print(i)

# Create New Data Frame with only important Data: id, pickup, dropoff, haversine, time. 

my_df = pd.DataFrame(tempdata)
my_df.to_csv('testlabelled.csv', index=False, header=["id", "time", "distance", "pickup", "dropoff"])

end = time.time()
print("completed in ", str(end-start))