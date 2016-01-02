from django.core.management.base import BaseCommand, CommandError
import numpy as np
from Intelligence.path.Path import *
import math
import pickle

class Command(BaseCommand):

	#args = '<directory and file containing features and files>'
    #help = 'create distance matrices for RGB and HSV histograms'    

	def handle(self, *args, **options) :

		if(len(args) == 0):
			raise CommandError("Please provide some arguments")
		images_number = args[0]
		kernel_type = args[1]
		print images_number
		#clusters_number = (int(math.ceil(math.sqrt(math.sqrt(float(images_number))))))**2
		clusters = pickle.load(open(DATA_PATH+'clusters-to-datapoints-' + kernel_type + '-' + str(images_number)))
		data = np.load(DATA_PATH + 'kernel-' + kernel_type + '-'+str(images_number)+'.npy')
		for i in range(len(clusters.keys())):
		#for i in range(2):

			images_for_cluster_i = clusters[i]
			temp_dist_mat = (data[images_for_cluster_i,:])[:, images_for_cluster_i]
			np.save(DATA_PATH + 'kernel-' + kernel_type + "-" + str(i) + '-' + str(images_number), temp_dist_mat)

'''
	def __init__(self, images_number_total, kernel_type):
		self.DATA_PATH = '/data/Imse/Data/'
		self.images_number = images_number_total
		self.clusters_number = (int(math.ceil(math.sqrt(math.sqrt(self.images_number)))))**2
		self.kernel_type = 'cl'
		self.clusters = pickle.load(open(DATA_PATH+'clusters-to-datapoints-' + self.kernel_type + '-' + str(images_number_total)))
		self.data = numpy.load(DATA_PATH + 'kernel-' + self.kernel_type + '-'+str(IMAGENUM)+'.npy')

	def extract_distance_matrix(self):

		#for i in range(len(self.clusters.keys())):
		for i in range(2):

			images_for_cluster_i = self.clusters[i]
			temp_dist_mat = (data[images_for_cluster_i,:])[:, images_for_cluster_i]
			numpy.save('/data/Imse/Data/kernel-' + self.kernel_type + "-" + i + '-' + str(images_number), temp_dist_mat)

'''