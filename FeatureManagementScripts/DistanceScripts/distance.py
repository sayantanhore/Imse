import os
import sys
import numpy as np
import math

'''
We're calculating the distance between two feature vectors here for IMSE.
The user has chosen to score (between 0.0-1.0) n images, each represented by a 1*4096 feature vector.
We estimate the importance of each feature by multiplying the inverted log distance by the score 
(so importance of a feature should be closer to the average score of the images, given how close they
are to each other).

per feature:
((1 - (chosen_variance / overall_variance)) * chosen_score) * activation_str
variance could also be std...
'''

nrofeatures = 4096
def calcDist(vecs, scores, allvariances): 
	distances = np.array() #[1:4096]
	chosen_variance = np.array()
	for feature in range(1,nrofaetures):
		chosen_variance = calcvariance(mean(vecs[:,feature]), vecs[:,feautre], scores) 
		distances[feature] = 1 - (chosen_variance / allvariances[feature])
	return distances
	
#this variance is to find out if feature is relevant given the k images (if their variance is low)
def calcvariance(featmean, vec, scores): #calculate euclidean between mean and each image feature, weighted by score
	variance = 0.0 #vec is k:1, one value for each image in the feature we're investigating now
	for image in vec:
		variance += scores[image] * math.sqrt(math.pow(vec[image]) - math.pow(featmean))
	return variance

def mean(vec): #for one feature, measure mean 
	return float(sum(vec)) / float(len(vec))

def distance(): #use this from outside
#if  __name__ =='__main__':
	data = open('vectors.txt','r').readlines() #[k:4096] images that user chose to score
	scores = open('scores.txt','r').readlines() #[k:1] scores the user gave each chosen image 
	vecs = np.asmatrix(data)
	scors = np.asmatrix(scores)
	allvariances = open('variances.txt','r').readlines() #[1:4096] features' variance over all images
	distances = calcDist(vecs, scors, allvariances)
	print distances
	
