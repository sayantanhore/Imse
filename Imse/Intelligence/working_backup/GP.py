import numpy
import pickle
import os
from scipy import spatial
import math
from time import gmtime, strftime
import logging

class GP(object):
    
    
    
    def __init__(self):
        '''
        do nothing -- just init '''
        #logger = logging.getLogger('django')
        
    # Gaussian process returns an index of chosen element from datapoints_predict
    def GP(self, datapoints_shown, feedback, datapoints_predict, data, time, sigma_n = 0.5):
        
        #logger.debug("Time :: " + str(time))
        print "Hello in GP"
        kernel = data
        ''' Before we were reading this kernel multiple times but this was very slow and now this part
        is done in global paremeters of views
        print 'Enter GP', strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        kernel = 1-numpy.load(self.file_path + 'features-rgb-kernel-'+str(self.images_number_total)+'.npy')
        '''        
        beta = math.sqrt(math.log(time))
        K = (kernel[datapoints_shown,:])[:,datapoints_shown]+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(datapoints_shown))))
        K_x = (kernel[datapoints_predict,:])[:,datapoints_shown]
        K_xx = kernel[datapoints_predict,datapoints_predict]+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(datapoints_predict))))
        temp = numpy.dot(K_x,numpy.linalg.inv(K))
        mean = numpy.dot(temp,feedback)
        sd = numpy.diag(K_xx - numpy.dot(temp,K_x.T))
        
        var = numpy.sqrt(sd)
        
        ucb = mean + 0.002 * var
        
        #pickle.dump(mean, open("/data/Imse/Data/mean-" + str(time), "w"))
        #numpy.savetxt("/data/Imse/Data/mean-" + str(time) + ".txt", mean)
        #numpy.savetxt("/data/Imse/Data/var-" + str(time) + ".txt", var)
        
        #pickle.dump(var, open("/data/Imse/Data/var-" + str(time), "w"))
        
        return ucb, mean


    