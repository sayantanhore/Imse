from django.core.management.base import BaseCommand, CommandError
from Intelligence.models import Image

from xml.dom.minidom import parse
import glob, os, math, copy
import numpy
from scipy import spatial
import pickle

import sys


class Command(BaseCommand) :
    args = '<directory and file containing features and files>'
    help = 'create distance matrices for RGB and HSV histograms'    
    def handle(self, *args, **options) :
        if len(args) != 1 :
            raise CommandError("Kernel command needs a number of images in the dataset")
        
        images_number = args[0]
        type = 'eh'
        
        '''data_distance = numpy.load('/home/fs/konyushk/fs/konyushk/Imse/Imse/Intelligence/Data/features-rgb-distance-'+str(images_number)+'.npy')
        cluster_approximation = numpy.load('/home/fs/konyushk/fs/konyushk/Imse/Imse/Intelligence/Data/cluster-approximation-rgb-'+str(images_number)+'.npy')
        
        clusters_to_data = data_distance[cluster_approximation,:]
        clusters_to_clusters = (data_distance[cluster_approximation,:])[:,cluster_approximation]

        kernel1 = numpy.hstack((data_distance, clusters_to_data.transpose()))
        kernel2 = numpy.hstack((clusters_to_data, clusters_to_clusters))
        kernel = numpy.vstack((kernel1, kernel2))
        
        print kernel
        
        numpy.save('/home/fs/konyushk/fs/konyushk/Imse/Imse/Intelligence/Data/features-rgb-kernel-'+str(images_number), kernel)
'''        
        #Kerenel version 2
        ''' for rgb data_distance = numpy.load(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Data/features-rgb-distance-'+str(images_number)+'.npy')))
        clusters = pickle.load(open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Data/cluster-to-datapoints-rgb'+str(images_number)))))
        0'''        
        data_distance = numpy.load('/data/Imse/Data/' + type + '_distances' + str(images_number)+'.npy')
        clusters = pickle.load(open('/data/Imse/Data/clusters-to-datapoints-' + type + "-"  + str(images_number)))
        
        images_number = int(images_number)      
        
        clusters_to_data = numpy.zeros((len(clusters), images_number))
        for cluster in clusters:
            clusters_to_data[int(cluster),:] = numpy.mean(data_distance[clusters[cluster],:], axis=0)
        #numpy.savetxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Data/test')), clusters_to_data) 
        
        clusters_to_clusters = numpy.zeros((len(clusters),len(clusters)))
        for cluster1 in clusters:
            for cluster2 in clusters:
                clusters_to_clusters[int(cluster1),int(cluster2)] = numpy.mean(((data_distance[clusters[cluster1],:])[:,clusters[cluster2]]))
        
        # Want to combine parts of the kernel:
        # clusters_to_clusters clusters_to_data
        # data_to_clusters data_to_data
        sigma_f = 0.5
        l = 0.5
        sigma_n = 0.8
        data_distance = (sigma_f**2)*numpy.exp(-(data_distance**2)/(2*(l**2)))#+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(data_distance))))
        clusters_to_data = (sigma_f**2)*numpy.exp(-(clusters_to_data**2)/(2*(l**2)))#+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(data_distance))))
        clusters_to_clusters = (sigma_f**2)*numpy.exp(-(clusters_to_clusters**2)/(2*(l**2)))
	    #print len(data_distance)
	    #print len(clusters_to_data)
	    #print len(clusters_to_clusters)        

        kernel1 = numpy.hstack((data_distance, clusters_to_data.transpose()))
        kernel2 = numpy.hstack((clusters_to_data, clusters_to_clusters))
        kernel = numpy.vstack((kernel1, kernel2))
        print len(kernel)
        ''' rgb kernel numpy.savetxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Data/R-features-'+str(images_number))), data_distance)'''
        
        numpy.save('/data/Imse/Data/kernel-' + type + "-"  + str(images_number), kernel)
        
        #numpy.save(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Data/features-rgb-kernel-'+str(images_number))), kernel)
        
        ''' Kernel version 1
        datafile = '/home/fs/konyushk/fs/konyushk/Imse/Imse/Intelligence/Data/features-rgb-'+str(images_number)+'.npy'
        clusterfile = '/home/fs/konyushk/fs/konyushk/Imse/Imse/Intelligence/Data/model-vectors-rgb-'+str(images_number)+'.npy'
        
        # Load datapoints and cluster centroids
        datapoints = numpy.load(datafile)
        images_number = len(datapoints)
        clusters = numpy.load(clusterfile)
        
        # Merge all the points into one matrix and then calculate the distance
        data = numpy.concatenate((datapoints, clusters)) 
        #Euclidean distance
        #distance = spatial.distance.cdist(data, data)
        print 'Start calculating distance...'
        distance = spatial.distance.cdist(data, data,'seuclidean')
        print 'Distances calculated!'
        print distance
        # For now let's try without Gaussian kernel
        kernel = distance
        #kernel = (sigma_f**2)*numpy.exp(-(distance**2)/(2*(l**2))) #+numpy.diag((sigma_n**2)*numpy.random.normal(1,0.1,(len(data))))
        numpy.save('/home/fs/konyushk/fs/konyushk/Imse/Imse/Intelligence/Data/features-rgb-kernel'+str(images_number), kernel)
        # Before we used this for user model, don't know, may be need some time, for instance for pure GP
        distance = distance[:images_number,:images_number]
        numpy.save('/home/fs/konyushk/fs/konyushk/Imse/Imse/Intelligence/Data/features-rgb-distance'+str(images_number), distance)'''
        
