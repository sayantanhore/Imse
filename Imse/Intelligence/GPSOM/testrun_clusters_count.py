# testrun.py - Simulating GPSOMmulti
# Author - Sayantan Hore

import GPSOM as gp
import numpy as np
import time
import csv


class TestRun(object):
	
	def __init__(self, exp_no, images_per_iteration, data_color):
	
		self.DATA_PATH = "/data/Imse/Data/"
		self.experiment_no = exp_no
		self.images_per_iteration = images_per_iteration
		self.IMAGENUM = 25000
		self.filename = "cluster_concentration_" + str(self.images_per_iteration) + ".csv"
		self.csv_writer = csv.writer(open(self.filename, "a+"))

		self.data_color = data_color
		#self.data_texture = data_texture
		#self.data_shape = data_shape

		# Initialize object

		predictor = gp.GPSOM(self.DATA_PATH, self.images_per_iteration, self.IMAGENUM, "None")

		record = [self.experiment_no, self.images_per_iteration]

		# Calling Firstround

		predictor.FirstRound()

		# Generate random feedback

		for i in range(10):
					
			feedback = self.feedback()

			# Calling Predict

			print "---------------------------------------------------------------------------Predict Start--------------------------------------------------------------------------------"
			start_at = time.time()
			clusters = predictor.Predict(feedback, self.data_color, self.data_texture, self.data_shape)
			#print "----------------------------------------------------------------------------Predict End---------------------------------------------------------------------------------"
			end_at = time.time()
			#record.append(end_at - start_at)
			record.append(i)
			record.append(clusters)
			print record
			#print "Time spent :: " + str(end_at - start_at) + " seconds"
			print "----------------------------------------------------------------------------Predict End---------------------------------------------------------------------------------"
		self.csv_writer.writerow(record)			
		del self.data_color
		del self.data_texture
		del self.data_shape

	def feedback(self):
		feedbacked_images = np.random.randint(0, 5)
		feedback = np.random.randint(0, 9, feedbacked_images).astype("float")/10
		filler = np.zeros(self.images_per_iteration - feedbacked_images)
		feedback = np.hstack((feedback, filler))
		np.random.shuffle(feedback)
		return feedback.tolist()
