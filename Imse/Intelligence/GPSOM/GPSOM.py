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
        self.setsize = images_number_iteration
        self.images_number = images_number_total
        self.first_sample_size = (int(math.ceil(math.sqrt(math.sqrt(self.images_number)))))**2
        self.category = category
        self.clusters = pickle.load(open(DATA_PATH+'clusters-to-datapoints-cl-' + str(images_number_total)))
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
        self.previouse_images = self.images_shown
        self.iteration += 1
        return self.images_shown
    
    def Predict(self, feedback, data):
        self.feedback = self.feedback + feedback
        
        # Get selected images
        i = 0
        for f in feedback:
            #if f!=0:
            self.selected_images.append(self.previouse_images[i])
            i += 1
        
        
        
        # What this method returns
        images = []
        
        # Copy all the values that will be used as they have to be modified only within iteration
        # Current training set with images and feedback and clusters assignments
        images_shown = copy.deepcopy(self.images_shown)
        feedback = copy.deepcopy(self.feedback)
        clusters = copy.deepcopy(self.clusters)
        
        
        clusters_names = range(self.images_number,self.images_number+self.clusters_number)
        print clusters_names
        
        while len(images)<self.setsize:

            # Changes for testing
            # ***************************************************************************************************************

            choosen_clusters = []

            # First choose a model vector chosen_model_vector
            # datapoints_predict - lines numbers of clusters in kernel
            datapoints_predict = clusters_names
            
            
            ucb, mean = self.gp.GP(images_shown, feedback, datapoints_predict, data, self.iteration)
            # This is a real cluster number   
               
            chosen_model_vector = clusters_names[ucb.argmax()]-self.images_number

            # Changes for testing
            # ***************************************************************************************************************

            choosen_clusters.append(chosen_model_vector)

            # From the chosen model vector choose data point
            datapoints_predict = clusters[chosen_model_vector]
            ucb, mean = self.gp.GP(images_shown, feedback, datapoints_predict, data, self.iteration)
            # Index of the chosen image in cluster assignment
            index_chosen_image = ucb.argmax() 
               
            
            # This is a real image number
            chosen_image = datapoints_predict[index_chosen_image]
            #predicted_image = (current_cluster_to_datapoint[chosen_model_vector])[index_chosen_image]
            #if chosen_image not in self.previouse_images:
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
