import pickle
import numpy
import random
import copy
import GP
import math
import time
from Intelligence.path.Path import *
import csv

class GPSOM(object):

    '''Program parameters'''
    #IMAGES_NUMBER = 1000

    def __init__(self, images_number_iteration, images_number_total, firstround_images_shown, data, clusters, category, file):
        self.setsize = images_number_iteration
        self.images_number = images_number_total
        self.clusters_number = (int(math.ceil(math.sqrt(math.sqrt(self.images_number)))))**2
        self.category = category
        self.data = data
        #self.clusters = pickle.load(open(DATA_PATH+'clusters-to-datapoints-cl-' + str(images_number_total)))
        self.clusters = clusters
        self.clusters_names = range(self.images_number, self.images_number + self.clusters_number)
        self.images_shown = firstround_images_shown
        self.previouse_images = []
        self.last_shown_images = None
        self.feedback = []
        self.exploration_rate = 0.2
        self.iteration = 0
        self.sub_iteration = 0
        self.gp = GP.GP(self.images_number, copy.deepcopy(self.images_shown), self.data, self.exploration_rate)
        self.selected_images = []
        self.chosen_model_vector = None
        self.index_chosen_image = None
        self.chosen_image = None
        self.pseudo_feedback = None
        self.bulk_predicted = False
        self.csv_file = csv.writer(file, delimiter = ",")

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


    def Predict(self, feedback, accepted, num_predictions = 1):
        record = [self.iteration, self.sub_iteration, self.images_shown, feedback, self.exploration_rate]
        if num_predictions == 1:
            print "Num Predictions :: " + str(num_predictions)
            print "Bulk prediction :: " + str(self.bulk_predicted)
            if self.bulk_predicted == True:
                self.bulk_predicted = False
                time_start = time.time()
                image_to_return = self.Predict_n(feedback, True)
                time_end = time.time()
            else:
                time_start = time.time()
                image_to_return = self.Predict_n(feedback, accepted)
                time_end = time.time()
            self.sub_iteration += 1
            record.append([image_to_return])
            record.append(time_end - time_start)
            self.csv_file.writerow(record)
            return [image_to_return]
        else:
            print "Num Predictions :: " + str(num_predictions)
            self.feedback = self.feedback + feedback
            images_to_return = []
            feedback = []
            img_counter = num_predictions
            time_start = time.time()
            while img_counter > 0:
                if img_counter == num_predictions:
                    if self.bulk_predicted == True:
                        self.bulk_predicted = False
                        images_to_return.append(self.Predict_n(feedback, True))
                    else:
                        images_to_return.append(self.Predict_n(feedback, accepted))
                else:
                    feedback.append(self.pseudo_feedback)
                    images_to_return.append(self.Predict_n(feedback, True))
                img_counter -= 1
            print "No of images returned :: " + str(img_counter)
            time_end = time.time()
            self.bulk_predicted = True
            self.iteration += 1
            self.sub_iteration = 0
            record.append(images_to_return)
            record.append(time_end - time_start)
            self.csv_file.writerow(record)
            return images_to_return

    def Predict_n(self, feedback, accepted = False):

        print "In Predict_n"
        if accepted == True:
            self.images_shown.append(self.chosen_image)
            #self.feedback.append(self.pseudo_feedback)

            # Delete

            self.clusters[self.chosen_model_vector] = numpy.delete(self.clusters[self.chosen_model_vector], self.index_chosen_image)

            if len(self.clusters[self.chosen_model_vector]) == 0:
                del self.clusters[self.chosen_model_vector]
                index_chosen_model_vector = list(self.clusters_names).index(self.chosen_model_vector + self.images_number)
                self.clusters_names = numpy.delete(self.clusters_names, index_chosen_model_vector)
        #self.feedback = feedback
        print "Feedback Vector :: " + str(len(self.feedback))
        print self.feedback
        datapoints_predict = self.clusters_names


        if accepted == True:
            print "Accepted"
            ucb, mean = self.gp.GP(self.feedback + feedback, datapoints_predict, "clusters", self.iteration, [self.chosen_image])
        else:
            print "Not accepted"
            ucb, mean = self.gp.GP(self.feedback + feedback, datapoints_predict, "clusters", self.iteration)
        print "Cluster selected"
        self.chosen_model_vector = self.clusters_names[ucb.argmax()]-self.images_number
        print "Hello chosen model vector :: " + str(self.chosen_model_vector)
        datapoints_predict = self.clusters[self.chosen_model_vector]
        ucb, mean = self.gp.GP(self.feedback + feedback, datapoints_predict, "images", self.iteration)

        self.index_chosen_image = ucb.argmax()
        self.chosen_image = datapoints_predict[self.index_chosen_image]
        print "chosen image " + str(self.chosen_image)
        self.pseudo_feedback = float("{0:.2f}".format(mean[self.index_chosen_image]))
        print "Pseudo Feedback :: " + str(self.pseudo_feedback)
        print "Checking before returning"
        return self.chosen_image

    def Predict2(self, feedback, data):
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