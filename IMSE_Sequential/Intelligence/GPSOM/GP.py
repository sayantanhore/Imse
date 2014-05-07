import numpy
import pickle
import os
from scipy import spatial
import math
from time import gmtime, strftime
import logging

class GP(object):
    
    
    
    def __init__(self, images_number_total, firstround_images_shown, data):
        
        self.images_number = images_number_total
        self.data = data
        self.datapoints_shown = firstround_images_shown
        self.datapoints_predict = None
        self.K = None
        self.K_inv = None
        self.K_x_clusters = None
        self.K_xx_clusters = None
        self.K_x_images = None
        self.K_xx_images = None
        self.temp = None
        
        self.sd_clusters = None
        
    # Gaussian process returns an index of chosen element from datapoints_predict
    def GP(self, feedback, datapoints_predict, called_for, time, accepted_image = None, sigma_n = 0.5):
        kernel = self.data
        mean = None
        sd = None

        if called_for == "clusters":
            if self.K is None:
                self.K = (kernel[self.datapoints_shown,:])[:,self.datapoints_shown]+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(self.datapoints_shown))))
                self.K_x_clusters = (kernel[datapoints_predict,:])[:,self.datapoints_shown]
                self.K_xx_clusters = (kernel[datapoints_predict, :])[:, datapoints_predict] + numpy.diag((sigma_n**2) * numpy.random.normal(1, 0.1, (len(datapoints_predict))))
                self.K_inv = numpy.linalg.inv(self.K)
                self.temp = numpy.dot(self.K_x_clusters, self.K_inv)
                self.sd_clusters = numpy.diag(self.K_xx_clusters - numpy.dot(self.temp, self.K_x_clusters.T))
                self.datapoints_predict = datapoints_predict
                
            elif not accepted_image is None:
                K = (kernel[accepted_image, :])[:, self.datapoints_shown]
                self.K = numpy.vstack((self.K, K))
                K = numpy.append(K, 0.25 + (sigma_n**2) * numpy.random.normal(1, 0.1, 1))[numpy.newaxis, :].T
                
                self.K = numpy.hstack((self.K, K))
                
                K_x_clusters = (kernel[datapoints_predict, :])[:, accepted_image]
                
                
                if len(datapoints_predict) < len(self.datapoints_predict):
                    deleted_cluster = list(set(self.datapoints_predict) - set(datapoints_predict))
                    #print deleted_cluster
                    #print self.images_number
                    deleted_cluster = deleted_cluster[-1] - self.images_number
                    #print deleted_cluster
                    self.K_x_clusters = numpy.delete(self.K_x_clusters, deleted_cluster, 0)
                    self.K_xx_clusters = numpy.delete(self.K_xx_clusters, deleted_cluster, 0)
                    #print self.K_x_clusters.shape
                    self.K_xx_clusters = numpy.delete(self.K_xx_clusters, deleted_cluster, 1)
                    self.datapoints_predict = datapoints_predict
                
                self.K_x_clusters = numpy.hstack((self.K_x_clusters, K_x_clusters))
                
                self.datapoints_shown.append(accepted_image[0])
                
                self.K_inv = numpy.linalg.inv(self.K)
                
                self.temp = numpy.dot(self.K_x_clusters, self.K_inv)
                
                self.sd_clusters = numpy.diag(self.K_xx_clusters - numpy.dot(self.temp, self.K_x_clusters.T))
                
            mean = numpy.dot(self.temp, feedback)
            
            sd = self.sd_clusters
            
            print "K-Shape :: " + str(self.K.shape)
            
        elif called_for == "images":
            self.K_x_images = (kernel[datapoints_predict, :])[:, self.datapoints_shown]
            self.K_xx_images = (kernel[datapoints_predict, :])[:, datapoints_predict] + numpy.diag((sigma_n**2) * numpy.random.normal(1, 0.1, (len(datapoints_predict))))
            temp = numpy.dot(self.K_x_images, self.K_inv)
            mean = numpy.dot(temp, feedback)
            sd = numpy.diag(self.K_xx_images - numpy.dot(temp, self.K_x_images.T))
            
        
        var = numpy.sqrt(sd)
        ucb = mean + 0.002 * var
        
        return ucb, mean


    