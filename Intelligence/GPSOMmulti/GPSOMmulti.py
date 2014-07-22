import pickle
import numpy
import random
import copy
import GP
import math
from scipy import stats
from Intelligence.path.Path import *

class GPSOMmulti(object):
    
    '''Program parameters'''
    #IMAGES_NUMBER = 1000
    
    def __init__(self, images_number_iteration, images_number_total, category):
        self.setsize = images_number_iteration
        self.images_number = images_number_total
        self.clusters_number = (int(math.ceil(math.sqrt(math.sqrt(self.images_number)))))**2
        self.category = category
        self.clusters_color = pickle.load(open(DATA_PATH + 'clusters-to-datapoints-cl-' + str(images_number_total)))
        self.clusters_texture = pickle.load(open(DATA_PATH + 'clusters-to-datapoints-ht-' + str(images_number_total)))
        self.clusters_shape = pickle.load(open(DATA_PATH + 'clusters-to-datapoints-eh-' + str(images_number_total)))
        
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
        if(self.category == "None"):
            clusters_per_group = int(math.ceil(self.clusters_number / self.setsize))
        else:
            clusters_per_group = 2 * int(math.ceil(self.clusters_number / self.setsize))
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
            r = random.randint(0, len(self.clusters_color[cluster])-1)
            images.append(self.clusters_color[cluster][r])
            
        '''
        chosen_clusters = chosen_clusters[:self.setsize]
        images = []
        for c in chosen_clusters:
            r = random.randint(0, len(self.clusters_color[c])-1)
            images.append(self.clusters_color[c][r])
        '''
        
        # Appending images from category selected by user to the returned random first round
        
        if(self.category != "None"):
            tags = pickle.load(open(DATA_PATH + "tag_to_img_" + str(self.images_number)))
            candidates = tags[self.category]
            images_from_selected_category = random.sample(candidates, self.setsize / 2)
            images.extend(images_from_selected_category)
        random.shuffle(images)
            
        self.images_shown = images
        self.previouse_images = images
        self.iteration += 1
        
        
        return images
    
    def Predict(self, feedback, data_color, data_texture, data_shape):
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
        
        
        
        #--------------------LEARN KERNEL COMBINATION---------------
        
        datapoints_predict = images_shown
        data = data_color
        smth, mean_color = self.gp.GP(images_shown, feedback, datapoints_predict, data, self.iteration)
        mse_color = numpy.mean((feedback-mean_color)**2)
        pearson_color = stats.pearsonr(feedback,mean_color)[0]
        
        datapoints_predict = images_shown
        data = data_texture
        smth, mean_texture = self.gp.GP(images_shown, feedback, datapoints_predict, data, self.iteration)
        mse_texture = numpy.mean((feedback-mean_texture)**2)
        pearson_texture = stats.pearsonr(feedback,mean_texture)[0]
         
        datapoints_predict = images_shown
        data = data_shape
        smth, mean_shape = self.gp.GP(images_shown, feedback, datapoints_predict, data, self.iteration)
        mse_shape = numpy.mean((feedback-mean_shape)**2)
        pearson_shape = stats.pearsonr(feedback,mean_shape)[0]
        
        
        
        c_w = 1.0*(mse_texture+mse_shape)/(mse_color+mse_texture+mse_shape)
        t_w = 1.0*(mse_color+mse_shape)/(mse_color+mse_texture+mse_shape)
        s_w = 1.0*(mse_color+mse_texture)/(mse_color+mse_texture+mse_shape)
        
        c_p = 1.0*pearson_color/(pearson_color+pearson_texture+pearson_shape)
        t_p = 1.0*pearson_texture/(pearson_color+pearson_texture+pearson_shape)
        s_p = 1.0*pearson_shape/(pearson_color+pearson_texture+pearson_shape)


	   # Start (Added by Sayantan 24/07/13)
        
        weightDictionary = {"colorWeight" : c_w, "textureWeight" : t_w, "shapeWeight" : s_w, "colorPearson" : c_p, "texturePearson" : t_p, "shapePearson" : s_p}
        
        # End
            
        
        print '----------------------------'
        #print 'feedback = ', feedback
        print 'COLOR:'
        #print 'PREDICTION = ', mean_color
        print 'PEARSON = ', pearson_color
        #print 'MSE = ', c_w
        print 'TEXTURE:'
        #print 'PREDICTION = ', mean_texture
        print 'PEARSON = ', pearson_texture
        #print 'MSE = ', t_w
        print 'SHAPE:'
        #print 'PREDICTION = ', mean_shape
        print 'PEARSON = ', pearson_shape
        #print 'MSE = ', s_w
        
        print '----------------------------'



        #-----------------------------------------------------------
        
        
        
        clusters_color = copy.deepcopy(self.clusters_color)
        clusters_texture = copy.deepcopy(self.clusters_texture)
        clusters_shape = copy.deepcopy(self.clusters_shape)
        
        clusters_names_color = range(self.images_number,self.images_number+self.clusters_number)
        clusters_names_texture = range(self.images_number,self.images_number+self.clusters_number)
        clusters_names_shape = range(self.images_number,self.images_number+self.clusters_number)
        
        tries = []
         
        while len(images)<self.setsize:
            
            # First choose a model vector chosen_model_vector
            # datapoints_predict - lines numbers of clusters in kernel
            datapoints_predict = clusters_names_color
            ucb_color, mean = self.gp.GP(images_shown, feedback, datapoints_predict, data_color, self.iteration)
            datapoints_predict = clusters_names_texture
            ucb_texture, mean = self.gp.GP(images_shown, feedback, datapoints_predict, data_texture, self.iteration)
            datapoints_predict = clusters_names_shape
            ucb_shape, mean = self.gp.GP(images_shown, feedback, datapoints_predict, data_shape, self.iteration)
            # This is a real cluster number   
               
               
            chosen_model_vector_color = clusters_names_color[ucb_color.argmax()]-self.images_number
            chosen_model_vector_texture = clusters_names_texture[ucb_texture.argmax()]-self.images_number
            chosen_model_vector_shape = clusters_names_shape[ucb_shape.argmax()]-self.images_number
            
            # From the chosen model vector choose data point
            datapoints_predict_color = clusters_color[chosen_model_vector_color]
            datapoints_predict_texture = clusters_texture[chosen_model_vector_texture]
            datapoints_predict_shape = clusters_shape[chosen_model_vector_shape]
            
            datapoints_predict = numpy.concatenate((datapoints_predict_color, datapoints_predict_texture, datapoints_predict_shape))
            #print 'datapoints_predict = ', datapoints_predict
            #print 'tries = ', tries
            for i in tries:
                index = numpy.where(datapoints_predict==i)
                datapoints_predict = numpy.delete(datapoints_predict, index[0])            
            #print 'datapoints_predict new = ', datapoints_predict
            #!!!!!!!!!!!!!!!!!1 The learning should be here !!!!!!!!!!!!!!!!!!!#
            #data = c_p*data_color + t_p*data_texture + s_p*data_shape
                         
            ucb_c, mean_c = self.gp.GP(images_shown, feedback, datapoints_predict, data_color, self.iteration)
            ucb_t, mean_t = self.gp.GP(images_shown, feedback, datapoints_predict, data_texture, self.iteration)
            ucb_s, mean_s = self.gp.GP(images_shown, feedback, datapoints_predict, data_shape, self.iteration)
            # Index of the chosen image in cluster assignment
            ucb = c_p*ucb_c + t_p*ucb_t + s_p*ucb_s
            #print c_p,'*',ucb_c, '+', t_p,'*',ucb_t, '+', s_p,'*',ucb_s
            mean = c_p*mean_c + t_p*mean_t + s_p*mean_s
            
            index_chosen_image = ucb.argmax() 
            
            #print 'chosen image = ', index_chosen_image
            
            
            
            
            # This is a real image number
            chosen_image = datapoints_predict[index_chosen_image]
            tries.append(chosen_image)
            #predicted_image = (current_cluster_to_datapoint[chosen_model_vector])[index_chosen_image]
            #print chosen_image
            if chosen_image not in self.previouse_images and chosen_image not in images_shown:
                if chosen_image not in self.selected_images:
                    if chosen_image not in images:
                        images.append(chosen_image)
                        # To sample next images we add fake feedback
                        images_shown.append(chosen_image)
                        feedback.append(mean[index_chosen_image])
            # Delete the chosen image from the current copy of cluster_to_datapoints in order not to sample it again
            #print '#color=', len(datapoints_predict_color)
            #print '#texture=', len(datapoints_predict_texture)
            #print '#shape=', len(datapoints_predict_shape)
            if index_chosen_image<len(datapoints_predict_color):
                print 'c'
                '''clusters_color[chosen_model_vector_color] = numpy.delete(clusters_color[chosen_model_vector_color],index_chosen_image)
                # if we have deleted all datapoints from the cluster, delete the cluster
                if len(clusters_color[chosen_model_vector_color])==0:
                    del clusters_color[chosen_model_vector_color]
                    index_chosen_model_vector_color = list(clusters_names_color).index(chosen_model_vector_color+self.images_number)
                    clusters_names_color = numpy.delete(clusters_names_color, index_chosen_model_vector_color)'''
            else:
                if (index_chosen_image-len(datapoints_predict_color))<len(datapoints_predict_texture):
                    print 't'
                    '''clusters_texture[chosen_model_vector_texture] = numpy.delete(clusters_texture[chosen_model_vector_texture],(index_chosen_image-len(datapoints_predict_color)))
                    if len(clusters_texture[chosen_model_vector_texture])==0:
                        del clusters_texture[chosen_model_vector_texture]
                        index_chosen_model_vector_texture = list(clusters_names_texture).index(chosen_model_vector_texture+self.images_number)
                        clusters_names_texture = numpy.delete(clusters_names_texture, index_chosen_model_vector_texture)    '''
                else:       
                    print 's'         
                    '''clusters_shape[chosen_model_vector_shape] = numpy.delete(clusters_shape[chosen_model_vector_shape],(index_chosen_image-len(datapoints_predict_color)-len(datapoints_predict_texture)))
                    if len(clusters_shape[chosen_model_vector_shape])==0:
                        del clusters_shape[chosen_model_vector_shape]
                        index_chosen_model_vector_shape = list(clusters_names_shape).index(chosen_model_vector_shape+self.images_number)
                        clusters_names_shape = numpy.delete(clusters_names_shape, index_chosen_model_vector_shape)'''
            
        self.images_shown = self.images_shown + images
        self.iteration += 1
        self.previouse_images = images
        return images, weightDictionary
