from django.core.management.base import BaseCommand, CommandError
#from Imse.Intelligence.models import Image
import numpy as np

class Command(BaseCommand):
    args = '<number of images>'
    help = 'calculating cluster centroids'
    
    def handle(self, *args, **options):
        if len(args) != 1:
            raise CommandError("Please specify the number of images")
        
        DATA_PATH = "/data/Imse/Data/"
        
        images_number = args[0]
        
        kernel_type = "cl"
        
        # Read cluster_to_datapoints
        
        clusters_to_datapoints = np.load(DATA_PATH + "clusters-to-datapoints-cl-" + str(images_number))
        
        # Read the feature file for color
        
        cl_features = np.load(DATA_PATH + "cl" + str(images_number) + ".npy")
        
        no_of_clusters = len(clusters_to_datapoints.keys())
        
        no_of_features = cl_features.shape[1]
        
        centroids = np.zeros((no_of_clusters, no_of_features))
        
        for i in range(no_of_clusters):
            centroids[i, :] = np.mean(cl_features[clusters_to_datapoints[i], :], axis = 0)
            
        np.save(DATA_PATH + "cl-centroids-" + str(images_number) + ".npy", centroids)