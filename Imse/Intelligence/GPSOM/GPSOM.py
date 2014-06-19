import pickle
import numpy as np
import random
import copy
import gp_cuda
import math
from Intelligence.path.Path import *

class GPSOM(object):
    
    '''Program parameters'''
    #IMAGES_NUMBER = 1000
    
    def __init__(self, images_number_iteration, images_number_total, category):
        self.image_features = np.asfarray(np.load("../../data/cl25000.npy"), dtype="float32")
        self.shown_images_mask = np.ones(self.image_features.shape[0], dtype=bool)
        self.setsize = images_number_iteration
        self.images_number = images_number_total
        self.first_sample_size = (int(math.ceil(math.sqrt(math.sqrt(self.images_number)))))**2
        self.images_shown = []
        self.previous_images = []
        self.feedback = []
        self.iteration = 0
        self.gp = None
        self.selected_images = []
    
    def FirstRound(self):
        """Pre-processing stage - sample first set of images
        Take random images"""
        self.images_shown = np.random.choice(self.images_number, self.first_sample_size, replace=False)
        for idx in self.images_shown:
            self.shown_images_mask[idx] = True
        self.previous_images = self.images_shown
        self.iteration += 1
        return self.images_shown
    
    def Predict(self, feedback, data):
        self.feedback = self.feedback + feedback
        if not self.gp:
            self.gp = gp_cuda.GaussianProcessGPU(self.image_features,
                                                 self.feedback,
                                                 self.images_shown)  # TODO: where do we get the indices and features?
        # What this method returns
        images = []
        # Copy all the values that will be used as they have to be modified only within iteration
        # Current training set with images and feedback and clusters assignments
        ucb, mean = self.gp.gaussianProcess()


        chosen_image_idx = ucb[self.shown_images_mask].argmax()
        if chosen_image not in images_shown:
            if chosen_image not in self.selected_images:
                images.append(chosen_image)
                # To sample next images we add fake feedback
                images_shown.append(chosen_image)
                feedback.append(mean[index_chosen_image])
        # Delete the chosen image from the current copy of cluster_to_datapoints in order not to sample it again
        clusters[chosen_model_vector] = numpy.delete(clusters[chosen_model_vector],index_chosen_image)
        # if we have deleted all datapoints from the cluster, delete the cluster
        if len(clusters[chosen_model_vector])==0:
            del clusters[chosen_model_vector]
            print chosen_model_vector+self.images_number
            index_chosen_model_vector = list(clusters_names).index(chosen_model_vector+self.images_number)
            print index_chosen_model_vector
            clusters_names = numpy.delete(clusters_names, index_chosen_model_vector)
        self.images_shown = self.images_shown + images
        self.iteration += 1
        self.previouse_images = images
        return images
        #return choosen_clusters
