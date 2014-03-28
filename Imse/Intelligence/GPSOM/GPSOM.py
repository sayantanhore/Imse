import pickle
import numpy
import random
import copy
import GP
import math
from Intelligence.path.Path import *

class GPSOM(object):
    
    '''Program parameters'''
    #IMAGES_NUMBER = 1000
    
    def __init__(self, images_number_iteration, images_number_total, category):
        self.setsize = images_number_iteration
        self.images_number = images_number_total
        self.clusters_number = (int(math.ceil(math.sqrt(math.sqrt(self.images_number)))))**2
        self.category = category
        self.clusters = pickle.load(open(DATA_PATH+'clusters-to-datapoints-cl-' + str(images_number_total)))
        self.images_shown = []
        self.previouse_images = []
        self.feedback = []
        self.iteration = 0
        self.gp = GP.GP()
        self.selected_images = []
    
    def FirstRound(self):
        
        '''Pre-processing stage - sample first set of images
        Take random images from different clusters 
        because they are the most remote ones'''
        
        
        chosen_clusters = numpy.arange(0,self.clusters_number)
        #numpy.random.shuffle(chosen_clusters)
        #if(self.category == "None"  and " " in self.category):
        clusters_per_group = int(math.ceil(self.clusters_number / self.setsize))
        #else:
            #clusters_per_group = 2 * int(math.ceil(self.clusters_number / self.setsize))
        cluster_counter = 0
        images = []
        while(cluster_counter < self.clusters_number -1):
            if((self.clusters_number - cluster_counter) >= clusters_per_group) and ((self.clusters_number - cluster_counter) < 2 * clusters_per_group):
                clusters_group = chosen_clusters[cluster_counter:]
                cluster_counter = self.clusters_number
            else:
                clusters_group = chosen_clusters[cluster_counter:cluster_counter + clusters_per_group]
                cluster_counter += clusters_per_group

            numpy.random.shuffle(clusters_group)
            cluster = clusters_group[0]
            r = random.randint(0, len(self.clusters[cluster])-1)
            images.append(self.clusters[cluster][r])
            
        '''
        chosen_clusters = chosen_clusters[:self.setsize]
        images = []
        for c in chosen_clusters:
            r = random.randint(0, len(self.clusters_color[c])-1)
            images.append(self.clusters_color[c][r])
        '''
        
        # Appending images from category selected by user to the returned random first round
        
        '''
        if(self.category != "None" and " " not in self.category):
            tags = pickle.load(open("/data/Imse/Data/tag_to_img_" + str(self.images_number)))
            candidates = tags[self.category]
            images_from_selected_category = random.sample(candidates, self.setsize / 2)
            images.extend(images_from_selected_category)
        '''
        random.shuffle(images)
        
            
        self.images_shown = images
        self.previouse_images = images
        self.iteration += 1
        
        
        return images
    
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
