import subprocess
import numpy as np
#import gp_cuda
#from Intelligence.path.Path import *

class GPSOM(object):
    
    '''Program parameters'''
    #IMAGES_NUMBER = 1000
    
    def __init__(self, images_number_iteration, images_number_total, firstround_images_shown, category):
        print "Inside GPSOM"
        #self.image_features = np.asfarray(np.load(DATA_PATH + "cl25000.npy"), dtype="float32")
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
        #self.feedback_indices.append(1)
        print "Before cuda initialization"
        print("After cuda initialization")
        # What this method returns
        images = []
        # Copy all the values that will be used as they have to be modified only within iteration
        # Current training set with images and feedback and clusters assignments
        print "Before calling gaussian process"

        feedback_str = '\t'.join(map(str, self.feedback)) + '\n'
        feedback_indices = np.arange(len(self.feedback), dtype=np.int)
        feedback_indices_str = '\t'.join(map(str, feedback_indices)) + '\n'
        print(feedback_str)
        print(feedback_indices_str)
        gp_process = None
        try:
            gp_process = subprocess.Popen('/home/lassetyr/programming/Imse/Imse/Intelligence/GPSOM/gp_cuda.py', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except Exception as e:
            print(e)
        print(gp_process)
        stdoutdata, stderrdata = gp_process.communicate(input=feedback_indices_str + feedback_indices_str)
        mean_variance = stdoutdata.strip().split('\n')
        print(len(mean_variance))
        mean = None
        try:
            mean = np.array(mean_variance[0].strip().split('\t'), dtype=np.float)
        except Exception as e:
            print(e)
        print(mean)
        variance = np.array(mean_variance[1].strip().split('\t'), dtype=np.float)
        print(variance)

        #mean, variance = gp_cuda.gaussian_process(self.image_features, self.feedback, self.feedback_indices)
        ucb = np.add(mean, variance)
        print "After calling gaussian process"
        chosen_image_indices = [ucb.argmax()]
        self.shown_images = self.shown_images + chosen_image_indices
        self.iteration += 1
        print(self.shown_images)
        return self.shown_images


