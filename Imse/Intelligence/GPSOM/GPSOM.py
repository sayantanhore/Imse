import numpy as np
import gp_cuda
from Intelligence.path.Path import *

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
    
    def FirstRound(self):
        """Pre-processing stage - sample first set of images
        Take random images"""
        self.feedback_indices = np.random.choice(self.images_number, self.first_sample_size, replace=False)
        for idx in self.feedback_indices:
            self.shown_images_mask[idx] = True
        self.iteration += 1
        return self.feedback_indices
    
    def Predict(self, feedback, num_predictions):
        print "Inside predict"
        self.feedback = self.feedback + feedback
        print "Before cuda initialization"
        print("After cuda initialization")
        # What this method returns
        images = []
        # Copy all the values that will be used as they have to be modified only within iteration
        # Current training set with images and feedback and clusters assignments
        print "Before calling gaussian process"
        mean, variance = gp_cuda.gaussian_process(self.image_features, self.feedback, self.feedback_indices)
        ucb = np.add(mean, variance)
        print "After calling gaussian process"
        if num_predictions == 1:
            chosen_image_indices = [ucb.argmax()]
        else:
            chosen_image_indices = sorted(ucb)[-num_predictions:]
        self.shown_images = self.shown_images + chosen_image_indices
        self.iteration += 1
        return self.shown_images


