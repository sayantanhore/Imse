import numpy as np
import copy
import gp_cuda as gp
from Intelligence.path.Path import *
import os, time, xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
from subprocess import Popen
from signal import SIGTERM
'''
def test2():
    return "This is test"
def gaussian_RPC():
    server = SimpleXMLRPCServer(("localhost", 8888))
    print "Configured"
    server.register_function(gp.test, "test")
    print "Listening at 8888"
    server.serve_forever()
def gaussian_RPC2():
    server = SimpleXMLRPCServer(("localhost", 8880))
    print "Configured"
    server.register_function(gp.gaussian_process, "gaussian_process")
    print "Listening at 8888"
    server.serve_forever()
'''

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
        self.selected_images = []
        print("Haha")
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

        #newpid = os.fork()
        #if newpid == 0:
            #gaussian_RPC()
        #else:
            #time.sleep(1)
        self.feedback = self.feedback + feedback
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
            mean, var = server_proxy.gp(self.feedback, self.shown_images.tolist())
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
        ucb = mean + var
        print("Hello hello")
        if num_predictions == 1:
            print "1 image"
            chosen_image_indices = np.array([ucb.argmax()])
            print "Chosen Image Incex :: " + str(type(chosen_image_indices))
            print chosen_image_indices[0]
        else:
            chosen_image_indices = sorted(ucb)[-num_predictions:]
            print "Chosen Image Incex :: " + str(chosen_image_indices)
        print "Image picked up"
        print(type(self.shown_images))
        #self.shown_images = self.shown_images + chosen_image_indices
        self.shown_images = np.append(self.shown_images, chosen_image_indices)
        print "Added to shown list"
        self.iteration += 1
        #os.kill(newpid, SIGTERM)
        print "Now kill process"
        #global p
        #p.send_signal(SIGTERM)
        print "Process killed"
        return chosen_image_indices


