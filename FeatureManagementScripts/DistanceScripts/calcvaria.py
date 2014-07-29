import numpy as np
import math
import os

def combineMatrix():
	features = np.zeros((4096,))
	#print 'perkel'
	for file in os.listdir('features/'):
		if file.endswith('res'): #cropdimnum.jpg.features. koska laiska	
			featvec = np.genfromtxt('%s%s' %('features/',file), dtype='float', delimiter=" ",skip_header=1)
			features = np.c_[features,featvec]
        features = features[:,1:25000]
        print features.shape
	return features

def calcVaria(matrix): #consider std
	return np.var(matrix, axis=1, dtype=np.float64)

if  __name__ =='__main__':
	print calcVaria(combineMatrix())


'''
import numpy as np
import math
import os

def combineMatrix():
	features = np.arange(25000*4096)
	features.shape=(25000,4096)
	for file in os.listdir('features/'):
		if file.endswith('res'): #cropdimnum.jpg.features. koska laiska
			featvec = np.genfromtxt('%s%s' %('features/',file), dtype='f8', delimiter=" ",skip_header=1)
			#np.concatenate((features, featvec), axis=1)
			features += featvec
	return features

def calcVaria(matrix): #consider std
	return np.var(matrix, axis=1, dtype=np.float64)

if  __name__ =='__main__':
	print calcVaria(combineMatrix())


'''
