import pandas as pd
import numpy as np
import math
import time
import csv


MODE = "test"
AVG_KPS = 0.00400644693588925  # Average speed of all routes.

"""

Iterate through training data and compute average speed for all taxi trips from / to each neighbourhood.

"""

start = time.time()

# TRAINING:
# Load Labelled set from CSV. For each From / To Destination add the KPS to an array. 

output = [[0 for i in range(7)] for i in range(100)]

data = pd.read_csv('trainlabelled.csv')


for index, row in data.iterrows():
	from_hood = int(row['pickup'])
	to_hood = int(row['dropoff'])
	output[from_hood][1] += row["distance"]
	output[from_hood][2] += row["trip_duration"]
	output[to_hood][4] += row["distance"]
	output[to_hood][5] += row["trip_duration"]

print("populated output array")

i = 0
for row in output:
	row[0] = i
	if row[2] != 0:
		row[3] = row[1] / row[2]
	else: row[3] = 0.004
	if row[5] != 0:
		row[6] = row[4] / row[5]
	else: row[6] = 0.004
	i += 1

my_df = pd.DataFrame(output)

my_df.to_csv('key.csv', index=False, header=["hood", "from_k", "from_s", "from_kps", "to_k", "to_s", "to_kps"])

print("completed in ", str(time.time()-start))