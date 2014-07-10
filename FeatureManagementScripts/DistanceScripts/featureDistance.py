import os
import sys 
import numpy as np 
import math 
''' per feature: ((1 - (chosen_variance / overall_variance)) 
* chosen_score) * activation_str variance could also be std... 
''' 
nrofeatures = 4096 
def calcDist(vecs, scores, allvariances):
	distances = np.zeros((nrofeatures)) #[1:4096]
	#chosen_variance = np.array([])
	for feature in range(1,nrofeatures):
		chosen_variance = calcVariance(mean(vecs[:,feature]), vecs[:,feature], scores)
		distances[feature] = 1 - (chosen_variance / allvariances[feature])
	return distances

def calcVariance(featmean, featvec, scores): #calculate euclidean between mean and each image feature, weighted by score
	variance = 0.0
	for image in range(0,len(featvec)):
		variance += scores[image] * math.sqrt(math.pow(float(featvec[image] - float(featmean)),2))
	return variance 

def mean(vec): #for one feature, measure mean
	return float(sum(vec)) / float(len(vec)) 

def fetchFeatures(imgs):
	vecs = []
	for img in imgs:
		filename = 'features/cropdim%s.jpg.features' %(img) #we assume 1-25000, not 0-24999, due to filename
		vecs.append(np.genfromtxt(filename, dtype='float', delimiter=","))
	print vecs
	return vecs


#def distance(): #use this from outside
if __name__ =='__main__':
	imgs = np.genfromtxt('chosenimages.txt', dtype='float', delimiter=",")
	scores = np.genfromtxt('scores.txt', dtype='float', delimiter=" ")
	variances = np.genfromtxt('variances.txt', dtype='float', delimiter=",")
	vecs = fetchFeatures(imgs)
	distances = calcDist(vecs, scores, variances)
	print distances
	#import distance as di di = reload(di)
