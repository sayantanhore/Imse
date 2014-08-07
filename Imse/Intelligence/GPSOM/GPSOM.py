import numpy as np
import copy
import gp_cuda as gp
from Intelligence.path.Path import *
import os, time, xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
from subprocess import Popen
from signal import SIGTERM
import csv
import time

class GPSOM(object):

    '''Program parameters'''
    #IMAGES_NUMBER = 1000

    def __init__(self, images_number_iteration, images_number_total, firstround_images_shown, category, file):
        print "Inside GPSOM"
        self.image_features = np.asfarray(np.load(DATA_PATH + "cl25000.npy"), dtype="float32")
        self.first_sample_size = images_number_iteration
        self.images_number = images_number_total
        self.shown_images = np.array(firstround_images_shown)
        self.feedback_indices = []
        self.feedback = []
        self.exploration_rate = 0
        self.iteration = 0  # TODO: use this to change the exploration/exploitation ratio
        self.sub_iteration = 0
        self.gp = None
        self.last_selected_image = None
        self.remaining_image_list = np.setdiff1d(np.array([i for i in range(images_number_total)]), self.shown_images)
        self.csv_file = csv.writer(file, delimiter = ",")

    def FirstRound(self):
        """Pre-processing stage - sample first set of images
        Take random images"""
        self.feedback_indices = np.random.choice(self.images_number, self.first_sample_size, replace=False)
        for idx in self.feedback_indices:
            self.shown_images_mask[idx] = True
        self.iteration += 1
        return self.feedback_indices

    def Predict(self, feedback, accepted, num_predictions = 1):
        record = [self.iteration, self.sub_iteration, self.shown_images, feedback, self.exploration_rate]
        print "Inside predict"
        print num_predictions

        #newpid = os.fork()
        #if newpid == 0:
            #gaussian_RPC()
        #else:
            #time.sleep(1)
        #self.feedback = feedback
        if accepted == True:
            self.shown_images = np.append(self.shown_images, np.array([self.last_selected_image]))
            self.remaining_image_list = np.setdiff1d(self.remaining_image_list, np.array([self.last_selected_image]))
        print "Before cuda initialization"
        '''
        if not self.gp:
            self.gp = gp_cuda.GaussianProcessGPU(self.image_features,
                                                self.feedback,
                                                self.shown_images)
        '''
        print("After cuda initialization")
        # What this method returns
        images = []
        # Copy all the values that will be used as they have to be modified only within iteration
        # Current training set with images and feedback and clusters assignments
        print "Before calling gaussian process"
        print "Go go"
        mean = None
        var = None
        try:
            print "Calling RPC now"
            server_proxy = xmlrpclib.ServerProxy("http://localhost:8888/")
            print "Server hit!!"
            txt = None
            print "1"
            print(type(feedback))
            print(type(self.shown_images))
            #print(server_proxy.system.methodHelp(gp))
            #mean, var = server_proxy.gp(self.image_features, self.feedback, self.shown_images)
            print("Feedback :::: " + str(self.feedback))
            print("Shown Images :::: " + str(self.shown_images))
            time_start = time.time()
            mean, var = server_proxy.gp(self.feedback + feedback, self.shown_images.tolist())
            time_end = time.time()
            mean = np.array(mean, dtype = "float32")
            var = np.array(var[0], dtype = "float32")
            print "2"
        except xmlrpclib.Fault as err:
            p.send_signal(SIGTERM)
            print(err.faultString)
        print type(mean)
        print mean.shape
        print mean
        print "3"
        print var.shape
        #mean, var = proxy.gaussian_process(self.image_features, self.feedback, self.shown_images)
        #var, mean = gp.gaussian_process(self.image_features, self.feedback, self.shown_images, debug=True)
        print "After calling gaussian process"
        ucb = mean + self.exploration_rate * np.sqrt(var)
        print("Hello hello")
        print "UCB type" + str(type(ucb))
        images_to_show = None
        if num_predictions == 1:
            print "1 image"
            chosen_image_indices = np.array([ucb.argmax()])
            print chosen_image_indices
            #self.last_selected_image = self.remaining_image_list[chosen_image_indices[0]]
            self.last_selected_image = self.remaining_image_list[chosen_image_indices[0]]
            print "Checking last selected image type :: "
            print type(self.last_selected_image)
            self.sub_iteration += 1
            images_to_show = self.remaining_image_list[chosen_image_indices.tolist()]
        else:
            print "Greater than 1"
            print(type(ucb))
            print (ucb.shape)
            #chosen_image_indices = ucb[0,:][-num_predictions:]
            chosen_image_indices = ucb.argsort()[-num_predictions:][::-1]
            print chosen_image_indices
            #self.shown_images = np.append(self.shown_images, chosen_image_indices)
            # Update the feedback
            self.feedback = self.feedback + feedback
            self.sub_iteration = 0
            self.iteration += 1
            images_to_show = self.remaining_image_list[chosen_image_indices.tolist()]
            self.shown_images = np.append(self.shown_images, images_to_show)
            self.remaining_image_list = np.setdiff1d(self.remaining_image_list, images_to_show)
        #self.shown_images = np.append(self.shown_images, chosen_image_indices)
        print("Checking before returning :: " + str(type(chosen_image_indices)))
        #images_to_show = self.remaining_image_list[chosen_image_indices.tolist()]
        #images_to_show = chosen_image_indices.tolist()
        record.append(images_to_show)
        record.append(time_end - time_start)
        self.csv_file.writerow(record)
        return images_to_show


