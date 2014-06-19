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
        self.first_sample_size = images_number_iteration
        self.images_number = images_number_total
        self.feedback_indices = []
        self.feedback = []
        self.iteration = 0  # TODO: use this to change the exploration/exploitation ratio
        self.gp = None
        self.selected_images = []
        self.chosen_model_vector = None
        self.index_chosen_image = None
        self.chosen_image = None
        self.pseudo_feedback = None
    
    def FirstRound(self):
        """Pre-processing stage - sample first set of images
        Take random images"""
        self.feedback_indices = np.random.choice(self.images_number, self.first_sample_size, replace=False)
        for idx in self.feedback_indices:
            self.shown_images_mask[idx] = True
        self.iteration += 1
        return self.feedback_indices
    
    def Predict(self, feedback, num_predictions):
        self.feedback = self.feedback + feedback
        if not self.gp:
            self.gp = gp_cuda.GaussianProcessGPU(self.image_features,
                                                 self.feedback,
                                                 self.feedback_indices)
        # What this method returns
        images = []
        # Copy all the values that will be used as they have to be modified only within iteration
        # Current training set with images and feedback and clusters assignments
        ucb, mean = self.gp.gaussianProcess()
        if num_predictions == 1:
            chosen_image_indices = [ucb.argmax()]
        else:
            chosen_image_indices = sorted(ucb)[-num_predictions:]
        self.feedback_indices = self.feedback_indices + chosen_image_indices
        self.iteration += 1
        return self.feedback_indices


