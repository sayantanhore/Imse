import numpy
import os
from scipy import spatial
import math
from time import gmtime, strftime

class GP(object):

    def __init__(self):
        '''
        do nothing -- just init '''

    # Gaussian process returns an index of chosen element from datapoints_predict
    def GP(self, datapoints_shown, feedback, datapoints_predict, data, random_K, random_K_xx, time = 1, sigma_n = 0.5):
        outfileprefix = 'output/' + str(len(feedback) - 12) + '_'
	print "Inside GP"
        kernel = data
        ''' Before we were reading this kernel multiple times but this was very slow and now this part
        is done in global paremeters of views
        print 'Enter GP', strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        kernel = 1-numpy.load(self.file_path + 'features-rgb-kernel-'+str(self.images_number_total)+'.npy')
        '''
        #beta = math.sqrt(math.log(time))
        #K = (kernel[datapoints_shown,:])[:,datapoints_shown]+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(datapoints_shown))))
        print datapoints_shown.shape
        print random_K.shape
        K = (kernel[datapoints_shown,:])[:,datapoints_shown]+numpy.diag(random_K)
        print "K computed"
	numpy.save(outfileprefix + "K.npy", K)
        K_x = (kernel[datapoints_predict,:])[:,datapoints_shown]
	numpy.save(outfileprefix + "K_x.npy", K_x)
        print "K_x computed"
        #K_xx = kernel[datapoints_predict,datapoints_predict]+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(datapoints_predict))))
        K_xx = kernel[datapoints_predict, datapoints_predict]+numpy.diag(random_K_xx)
        print "K_xx computed"
	numpy.save(outfileprefix + "K_inv.npy", numpy.linalg.inv(K))
        temp = numpy.dot(K_x,numpy.linalg.inv(K))
	numpy.save(outfileprefix + "temp.npy", temp)
        #print "Temp computed"
        mean = numpy.dot(temp,feedback)
        print "Mean computed"
	numpy.save(outfileprefix + "mean.npy", mean)
	numpy.save(outfileprefix + "diag.npy", numpy.dot(temp, K_x.T))
        print "K_xx shape"
        print K_xx.shape
        print numpy.diag(K_xx)
        var = numpy.abs(numpy.diag(K_xx - numpy.dot(temp,K_x.T)))
	numpy.save(outfileprefix + "var.npy", var)
        print "var computed"
        #ucb = mean + beta*numpy.sqrt(var)
        return mean, var
        # for demo only
        #print "For demo only"
        #return K, K_xx, mean, var



