import numpy as np
import os, time, xmlrpclib
import copy
from SimpleXMLRPCServer import SimpleXMLRPCServer
from subprocess import Popen
from signal import SIGTERM
import csv
import time
import logging
from django.conf import settings


class GP_GPU(object):

    '''Program parameters'''
    #IMAGES_NUMBER = 1000

    def __init__(self, no_of_images_to_show, no_of_images_total, firstround_images_shown):
        print("Enter >> 'GP_GPU__init__'")
        self.image_features = np.asfarray(np.load(settings.FEATURES), dtype="float32")
        self.no_of_images_to_show = no_of_images_to_show
        self.no_of_images_total = no_of_images_total
        #self.shown_images = np.array(firstround_images_shown)
        self.shown_images = []
        self.feedback_indices = []
        self.feedback = []
        self.exploration_rate = settings.EXPLORATION_RATE
        self.gp = None
        self.last_selected_image = None
        print("Exit << 'GP_GPU__init__'")

    def update(self, feedback, feedback_indices):
	print("Enter >> 'update'")
	# Update feedback and shown images
	self.feedback = self.feedback + feedback
	self.shown_images = self.shown_images + feedback_indices
	print("Exit << 'update'")

    def predict(self, feedback, feedback_indices, num_predictions, p):
	print("Enter >> 'predict'")
	mean = None
	var = None
	try:
		print("Calling RPC ...")
		try:
			f = open(settings.GP_PORT, "r")
			port = f.readline()
			f.close()
			server_proxy = xmlrpclib.ServerProxy(settings.GP_URL_INITIAL + port + "/")
		except Exception:
			print("ERROR :: PORT Undefined")
			print("Server hit!!")
		print "Server instance made"
		mean, var = server_proxy.gp(self.feedback + feedback, self.shown_images + feedback_indices)
		mean = np.array(mean, dtype = "float32")
		var = np.array(var, dtype = "float32")
	except xmlrpclib.Fault as err:
		p.send_signal(SIGTERM)
		print(err.faultString)
	ucb = mean + self.exploration_rate * np.sqrt(var)
	self.update(feedback, feedback_indices)
	chosen_image_indices = ucb.argsort()[::-1][-num_predictions:]
	print("Chosen image indices :: " + str(chosen_image_indices))
	remaining_image_list = np.setdiff1d(np.array([i for i in range(self.no_of_images_total)]), np.array(self.shown_images))
	images_to_show = remaining_image_list[chosen_image_indices.tolist()]
	print("Images to show :: " + str(images_to_show))
	print("Exit << 'predict'")
	return images_to_show

