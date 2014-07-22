import numpy as np
import copy
import gp_cuda as gp
from Intelligence.path.Path import *
import os, time, xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
from subprocess import Popen
from signal import SIGTERM

class GPSOM(object):

    '''Program parameters'''
    #IMAGES_NUMBER = 1000

    def __init__(self, images_number_iteration, images_number_total, firstround_images_shown, category):
        print "Inside GPSOM"
        self.image_features = np.asfarray(np.load(DATA_PATH + "cl25000.npy"), dtype="float32")
        self.first_sample_size = images_number_iteration
        self.images_number = images_number_total
        self.shown_images = np.array(firstround_images_shown)
        self.feedback_indices = []
        self.feedback = []
        self.iteration = 0  # TODO: use this to change the exploration/exploitation ratio
        self.gp = None
        self.last_selected_image = None
        self.remaining_image_list = np.setdiff1d(np.array([i for i in range(images_number_total)]), self.shown_images)
        #p = Popen(["python", "/ldata/IMSE/Imse/Imse/Intelligence/GPSOM/gp_cuda.py"])
        #time.sleep(0.8)

    def FirstRound(self):
        """Pre-processing stage - sample first set of images
        Take random images"""
        self.feedback_indices = np.random.choice(self.images_number, self.first_sample_size, replace=False)
        for idx in self.feedback_indices:
            self.shown_images_mask[idx] = True
        self.iteration += 1
        return self.feedback_indices

    def Predict(self, feedback, accepted, num_predictions = 1):
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
        mean = None
        var = None
        try:
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
            mean, var = server_proxy.gp(self.feedback + feedback, self.shown_images.tolist())
            mean = np.array(mean, dtype = "float32")
            var = np.array(var, dtype = "float32")
            print "2"
        except xmlrpclib.Fault as err:
            p.send_signal(SIGTERM)
            print(err.faultString)
        print type(mean)
        print mean.shape
        print mean
        print "3"
        #mean, var = proxy.gaussian_process(self.image_features, self.feedback, self.shown_images)
        #var, mean = gp.gaussian_process(self.image_features, self.feedback, self.shown_images, debug=True)
        print "After calling gaussian process"
        ucb = mean + 0.002 * var
        print("Hello hello")
        print "Num Predictions" + str(type(num_predictions))
        if num_predictions == 1:
            print "1 image"
            chosen_image_indices = np.array([ucb.argmax()])
            print "Chosen Image Incex :: " + str(type(chosen_image_indices))
            self.last_selected_image = chosen_image_indices[0]
        else:
            print "Greater than 1"
            print(ucb)
            #ucb.sort()
            print("UCB Sorted")
            print ucb.shape
            print "Just before slicing"
            print num_predictions
            #chosen_image_indices = ucb[0,:][-num_predictions:]
            chosen_image_indices = ucb[0,:].argsort()[-num_predictions:][::-1]
            print chosen_image_indices
            print "Chosen Image Incex :: " + str(type(chosen_image_indices))
            self.shown_images = np.append(self.shown_images, chosen_image_indices)
            # Update the feedback
            self.feedback = self.feedback + feedback
        print "Image picked up"
        print(type(self.shown_images))
        #self.shown_images = self.shown_images + chosen_image_indices
        #self.shown_images = np.append(self.shown_images, chosen_image_indices)
        print "Added to shown list"
        self.iteration += 1
        print("Checking before returning :: " + str(type(chosen_image_indices)))
        images_to_show = self.remaining_image_list[chosen_image_indices.tolist()]
        self.remaining_image_list = np.setdiff1d(self.remaining_image_list, images_to_show)
        return images_to_show


