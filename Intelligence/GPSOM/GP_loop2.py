import numpy
import pickle
import os
from scipy import spatial
import math
from time import gmtime, strftime
import logging


class GP(object):
    def __init__(self, images_number_total, data):
        #logger = logging.getLogger('django')
        self.images_number = images_number_total
        self.data = data
        self.datapoints_shown = None
        self.datapoints_predict = None
        self.K = None
        self.K_inv = None
        self.K_x_clusters = None
        self.K_xx_clusters = None
        self.K_x_images = None
        self.K_xx_images = None

    def GP(self, chosen_images, feedback, datapoints_predict, callfor, time, sigma_n = 0.5):
        print "Starting :::: "
        if self.datapoints_shown is not None:
            print chosen_images
            print self.datapoints_shown
        #logger.debug("Time :: " + str(time))

        kernel = self.data

        mean = None
        sd = None

        if callfor == "clusters":
            if self.K is None:
                print "Inside clusters - For the first time"
                self.datapoints_shown = chosen_images
                self.datapoints_predict = datapoints_predict
                #print len(self.datapoints_shown)
                self.K = (kernel[self.datapoints_shown, :])[:, self.datapoints_shown] + numpy.diag((sigma_n**2) * numpy.random.normal(1, 0.1, (len(self.datapoints_shown))))
                self.K_x_clusters = (kernel[datapoints_predict, :])[:, self.datapoints_shown]
                self.K_xx_clusters = (kernel[datapoints_predict, :])[:, datapoints_predict] + numpy.diag((sigma_n**2) * numpy.random.normal(1, 0.1, (len(datapoints_predict))))
            else:
                print "Inside clusters - Not the first time"
                #print len(self.datapoints_shown)
                #print "List length :: " + str(len(chosen_images))
                '''
                if chosen_images[-1] == -999:
                    print "Empty"
                    chosen_images = [self.datapoints_shown[-1]]
                    del self.datapoints_shown[-1]
                '''
                if len(chosen_images) > 1:
                    chosen_images = [chosen_images[-1]]
                K = (kernel[chosen_images, :])[:, self.datapoints_shown]
                #print self.K.shape
                #print K.shape
                self.K = numpy.vstack((self.K, K))
                #print self.K.shape
                #print K.shape
                K = numpy.append(K, 0.25 + (sigma_n**2) * numpy.random.normal(1, 0.1, 1))[numpy.newaxis, :].T
                #print K.shape
                self.K = numpy.hstack((self.K, K))
                K_x_clusters = (kernel[datapoints_predict, :])[:, chosen_images]
                #print "K_x shape :: " + str(K_x_clusters.shape)

                # Check whether self.K_xx is unchanged
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
                self.datapoints_shown.append(chosen_images[0])
            
            self.K_inv = numpy.linalg.inv(self.K)
            temp = numpy.dot(self.K_x_clusters, self.K_inv)
            print temp.shape
            print "Feedback size :: " + str(len(feedback))
            mean = numpy.dot(temp, feedback)
            sd = numpy.diag(self.K_xx_clusters - numpy.dot(temp, self.K_x_clusters.T))
            #print len(self.datapoints_shown)

        elif callfor == "images":
            print "Inside Images"
            #print len(self.datapoints_shown)
            # self.K unchanged
            self.K_x_images = (kernel[datapoints_predict, :])[:, self.datapoints_shown]
            self.K_xx_images = (kernel[datapoints_predict, :])[:, datapoints_predict] + numpy.diag((sigma_n**2) * numpy.random.normal(1, 0.1, (len(datapoints_predict))))

            temp = numpy.dot(self.K_x_images, self.K_inv)
            mean = numpy.dot(temp, feedback)
            sd = numpy.diag(self.K_xx_images - numpy.dot(temp, self.K_x_images.T))
            #print len(self.datapoints_shown)

        var = numpy.sqrt(sd)

        ucb = mean + 0.002 * var
        print "Before exit"
        #print len(self.datapoints_shown)
        return ucb, mean