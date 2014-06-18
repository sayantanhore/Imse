import Image
import ImageDraw
import cv
import copy
import numpy as np
from scipy import spatial

from Intelligence.path.Path import *

class LoadInitialImages(object):
	
	def __init__(self, colors):
		
          self.colors = colors.encode("utf-8").replace("[", "").replace("]", "").replace("\"", "").split(",")
          self.b = 8
          self.images_number = IMAGENUM

	def rgb_histogram(self, src):
         b = self.b
         B_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
         G_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
         R_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)   
         cv.Split(src, B_plane, G_plane, R_plane, None)
         planes = [B_plane, G_plane, R_plane]
         B_bins = b
         G_bins = b
         R_bins = b
         bins = [B_bins, G_bins, R_bins]
         B_ranges = [0, 225]
         G_ranges = [0, 225]
         R_ranges = [0, 225]
         ranges = [B_ranges, G_ranges, R_ranges]
         hist = cv.CreateHist(bins, cv.CV_HIST_ARRAY, ranges, 1)
         cv.CalcHist([cv.GetImage(i) for i in planes], hist)
         cv.NormalizeHist(hist, 1)
         return hist
	
	
	def load_images(self, no_of_images):
	  print "Inside load"	
          image = Image.new("RGB", (300, 300))
          imdraw = ImageDraw.Draw(image)
    		
          interval = 300 / len(self.colors)
    		
          for i in range(len(self.colors)):
              imdraw.rectangle([(0, i * interval), (300, (i+1) * interval)], fill = self.colors[i])
    		
          #image.save("/data/Imse/Data/123.jpg")
	  image.save(DATA_PATH + "123.jpg")
    	  print "Image created"
          image_in_cv = cv.LoadImageM(DATA_PATH + "123.jpg")
          hist = self.rgb_histogram(image_in_cv)
          hist_rgb = (np.array(hist.bins)).reshape(-1)
          
	  np.savetxt(DATA_PATH + "cl_dist_temp2.txt", hist_rgb.T)
    	  print "Hist Extracted"
          centroids = np.load(DATA_PATH + "cl-centroids-" + str(self.images_number) + ".npy")
	  print "Centroid picked"
          clusters_to_datapoints = np.load(DATA_PATH + "clusters-to-datapoints-cl-" + str(self.images_number))
          print "Cluster picked"
          hist_rgb = hist_rgb[np.newaxis]
          
          distances_from_centroids = spatial.distance.cdist(centroids, hist_rgb, "cityblock")
	  print "distances picked"
          shortest_cluster = np.argmin(distances_from_centroids)
          print "Shortest cluster selected"
          images_in_shortest_cluster = clusters_to_datapoints[shortest_cluster]
          #images_in_shortest_cluster = copy.deepcopy(clusters_to_datapoints[shortest_cluster])
          np.random.shuffle(images_in_shortest_cluster)
          images_to_show = []
          
          for i in range(no_of_images):
              images_to_show.append(images_in_shortest_cluster[i])
              images_in_shortest_cluster = np.delete(images_in_shortest_cluster, i)
              np.random.shuffle(images_in_shortest_cluster)
          print "Images extracted"
          if len(images_in_shortest_cluster) == 0:
              del images_in_shortest_cluster
	  print "Images fetched"
          return images_to_show, clusters_to_datapoints
