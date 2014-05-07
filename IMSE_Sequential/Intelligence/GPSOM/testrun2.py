# testrun.py - Simulating GPSOMmulti
# Author - Sayantan Hore

import GPSOMmulti as gp
import numpy as np
import time
import csv


class TestRun(object):
	
	def __init__(self, exp_no, images_per_iteration):
	
		self.DATA_PATH = "/data/Imse/Data/"
		self.experiment_no = exp_no
		self.images_per_iteration = images_per_iteration
		self.IMAGENUM = 25000
		self.filename = "times_multiprocessing_" + str(self.images_per_iteration) + ".csv"
		self.csv_writer = csv.writer(open(self.filename, "a+"))


		#self.data_color = np.load(self.DATA_PATH+ 'kernel-cl-'+str(self.IMAGENUM)+'.npy')
		#self.data_texture = np.load(self.DATA_PATH+ 'kernel-ht-'+str(self.IMAGENUM)+'.npy')
		#self.data_shape = np.load(self.DATA_PATH+ 'kernel-eh-'+str(self.IMAGENUM)+'.npy')

		# Initialize object

		predictor = gp.GPSOMmulti(self.images_per_iteration, self.IMAGENUM, "None")

		record = [self.experiment_no, self.images_per_iteration]

		# Calling Firstround

		predictor.FirstRound()

		# Generate random feedback

		for i in range(10):
			
			feedback = self.feedback()

			# Calling Predict

			print "---------------------------------------------------------------------------Predict Start--------------------------------------------------------------------------------"
			start_at = time.time()
			predictor.Predict(feedback)
			#print "----------------------------------------------------------------------------Predict End---------------------------------------------------------------------------------"
			end_at = time.time()
			record.append(end_at - start_at)
			print record
			#print "Time spent :: " + str(end_at - start_at) + " seconds"
			print "----------------------------------------------------------------------------Predict End---------------------------------------------------------------------------------"
		self.csv_writer.writerow(record)
		#del self.data_color
		#del self.data_texture
		#del self.data_shape

	def feedback(self):
		feedbacked_images = np.random.randint(0, 5)
		feedback = np.random.randint(0, 9, feedbacked_images).astype("float")/10
		filler = np.zeros(self.images_per_iteration - feedbacked_images)
		feedback = np.hstack((feedback, filler))
		np.random.shuffle(feedback)
		return feedback.tolist()
