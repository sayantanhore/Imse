from django.core.management.base import BaseCommand, CommandError
from Intelligence.models import Image

from xml.dom.minidom import parse
import glob, os, math, copy
import numpy
from scipy import spatial
import pickle
import scipy

import sys
#import cv

b = 8

def rgb_histogram(src):
	B_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
	G_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
	R_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)   
	cv.Split(src, B_plane, G_plane, R_plane, None)
	planes = [B_plane, G_plane, R_plane]
	B_bins = b
	G_bins = b
	R_bins = b
	bins = [B_bins, G_bins, R_bins]
	B_ranges = [0, 225]
	G_ranges = [0, 225]
	R_ranges = [0, 225]
	ranges = [B_ranges, G_ranges, R_ranges]
	hist = cv.CreateHist(bins, cv.CV_HIST_ARRAY, ranges, 1)
	cv.CalcHist([cv.GetImage(i) for i in planes], hist)
	cv.NormalizeHist(hist, 1)
	return hist

class Command(BaseCommand) :
	args = '<number of images>'
	help = 'homogeneous texture distance'
	
	def handle(self, *args, **options) :
		if len(args) != 1 :
			raise CommandError("Kernel command needs a number of images in the dataset")
		
		images_number = int(args[0])
		
		'''ht_data = numpy.loadtxt('/data/Imse/Data/ht'+str(images_number)+'.txt')
		ht_std = numpy.std(ht_data, 0)
		ht_normed = numpy.divide(ht_data, ht_std)
		distances = scipy.spatial.distance.cdist(ht_normed, ht_normed, 'cityblock')/len(ht_std)
		numpy.save('/data/Imse/Data/ht_distances'+str(images_number), distances)
		

		eh_data = numpy.loadtxt('/data/Imse/Data/eh'+str(images_number)+'.txt')
		eh_std = numpy.std(eh_data, 0)
		#eh_normed = numpy.divide(eh_data, eh_std)
		# Give more weight to global features
		eh_data[:,0:84]=eh_data[:,0:84]*0.8
		eh_data[:,85:150]=eh_data[:,85:150]*1.2
		distances = scipy.spatial.distance.cdist(eh_data, eh_data, 'cityblock')/(5.0*len(eh_std))
		numpy.save('/data/Imse/Data/eh_distances'+str(images_number), distances)

			
		Extract color histograms
		namesfile = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Data/imageset'+str(images_number) + '.nm'))
		names = [line.strip() for line in open(namesfile).readlines()]
		hists_rgb = []
		hists_rgb_values = numpy.zeros((images_number,b*b*b ))
		
		for pic in range(images_number):
		print pic
		image = cv.LoadImageM(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'media/' + names[pic])))
		hist = rgb_histogram(image)
		hists_rgb.append(hist)
		
		hists_rgb_values[pic,:] = (numpy.array(hist.bins)).reshape(-1) 
		
		numpy.save(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Data/cl'+str(images_number))), hists_rgb_values)
		'''
		cl_data = numpy.load('/data/Imse/Data/cl'+str(images_number)+'.npy')
		cl_std = numpy.std(cl_data, 0)
		#cl_normed = numpy.divide(cl_data, cl_std)
		distances = scipy.spatial.distance.cdist(cl_data, cl_data, 'cityblock')/len(cl_std)
		numpy.save('/data/Imse/Data/cl_distances'+str(images_number), distances)
		
