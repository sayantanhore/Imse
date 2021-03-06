import pickle
import numpy
import random
import copy
import GP
import math
import numpy as np
import scipy.spatial.distance as dist
from Intelligence.path.Path import *

class MLDist(object):

    '''Program parameters'''
    #IMAGES_NUMBER = 1000

    def __init__(self, images_number_iteration, images_number_total, firstround_images_shown, distancematrix, clusters, category):
        self.setsize = images_number_iteration
        self.images_number = images_number_total
        self.clusters_number = (int(math.ceil(math.sqrt(math.sqrt(self.images_number)))))**2
        self.category = category
        self.distance_matrix = distancematrix
        #self.clusters = pickle.load(open(DATA_PATH+'clusters-to-datapoints-cl-' + str(images_number_total)))
        self.clusters = clusters
        self.clusters_names = range(self.images_number, self.images_number + self.clusters_number)
        self.images_shown = firstround_images_shown
        self.image_features
        self.previous_images = []
        self.feedback = []
        self.iteration = 0
        self.gp = GP.GP(self.images_number, copy.deepcopy(self.images_shown), self.distance_matrix)
        self.selected_images = []
        self.chosen_model_vector = None
        self.index_chosen_image = None
        self.chosen_image = None
        self.pseudo_feedback = None

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

        random.shuffle(images)

        self.images_shown = images
        self.previous_images = images
        self.iteration += 1

        return images

    def Predict(self, feedback, accepted = False ):
        if accepted == True:
            self.images_shown.append(self.chosen_image)
            #self.feedback.append(self.pseudo_feedback)

            self.clusters[self.chosen_model_vector] = numpy.delete(self.clusters[self.chosen_model_vector], self.index_chosen_image)

            if len(self.clusters[self.chosen_model_vector]) == 0:
                del self.clusters[self.chosen_model_vector]
                index_chosen_model_vector = list(self.clusters_names).index(self.chosen_model_vector + self.images_number)
                self.clusters_names = numpy.delete(self.clusters_names, index_chosen_model_vector)
        self.feedback = feedback

        print "Feedback Vector :: " + str(len(self.feedback))
        print self.feedback
        datapoints_predict = self.clusters_names

        #insert distance metric learning here---------------
        #fetch feature vectors based on images_shown.
        #DO ONCE: for greater justice, combine all features into np.array
        features = np.zeros((4096,))
        featuresfolder = '/home/overfeat/features/'

        for i in range(1,25001):
            featvec = np.genfromtxt('cropdim%d.jpg.features' %(i), dtype = 'float', delimiter=" ",skip_header=1)
            features = np.c_[features,featvec]
        features = np.delete(features,0,1) #remove extra 0-column from the matrix
        #for filenro in self.images_shown:
        #    featvec = np.genfromtxt(featuresfolder+'cropdim%d.jpg.features' %(filenro), dtype = 'float', delimiter=" ",skip_header=1)
        #    features = np.c_[features,featvec]

        featurefile = 'feats_numpy_dump'
        np.save(featurefile, features)
        #END DO ONCE

        self.image_features = np.load(featurefile)
        #recalculate distance_matrix (was called 'data' in Sayantan's code)
        self.distance_matrix = self.GetRelDistances( features )
        self.GP.data = self.distance_matrix #poorly named, and you should add the GP again to MLDists folder
        #end insert----------------------------------------

        if accepted == True:
            ucb, mean = self.gp.GP(self.feedback, datapoints_predict, "clusters", self.iteration, [self.chosen_image])
        else:
            ucb, mean = self.gp.GP(self.feedback, datapoints_predict, "clusters", self.iteration)

        self.chosen_model_vector = self.clusters_names[ucb.argmax()]-self.images_number
        print "Hello chosen model vector :: " + str(self.chosen_model_vector)
        datapoints_predict = self.clusters[self.chosen_model_vector]

        ucb, mean = self.gp.GP(self.feedback, datapoints_predict, "images", self.iteration)

        self.index_chosen_image = ucb.argmax()
        self.chosen_image = datapoints_predict[self.index_chosen_image]
        print "chosen image " + str(self.chosen_image)

        self.pseudo_feedback = float("{0:.2f}".format(mean[self.index_chosen_image]))

        return self.chosen_image

    def GetRelDistances(self): #got code from Lasse. fixed. UNTESTED
        #Feature significance calculation
        chosen_images = np.array([self.image_features[i] for i in self.index_chosen_image])
        weights = np.expand_dims(self.feedback, axis=1)
        weighted_mean = np.sum(np.multiply(weights, chosen_images),axis=0)/np.sum(weights)#np.mean(np.multiply(weights, chosen_images), axis=0)
        weighted_variance = np.sum(np.square(np.multiply(weights, np.subtract(chosen_images, weighted_mean))), axis=0)
        significance = np.subtract(1, np.divide(weighted_variance, self.variance))
        significance[significance < 0.0] = 0.0 #this normalizes situations where chosen variance would had been greter than overall variance (which is possible)

        # Distance calculation
        signif_chosen_images = np.multiply(significance, chosen_images)
        self.distance_matrix = dist.cdist(signif_chosen_images, signif_chosen_images) #passing chosen_images * chosen_images distance matrix now. ask if needed to pass chosein_images * all_images
