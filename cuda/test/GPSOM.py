import gp_cuda
import numpy as np

class GPSOM(object):
    
    '''Program parameters'''
    #IMAGES_NUMBER = 1000
    
    def __init__(self):
        self.image_features = None
        self.feedback_indices = []
        self.feedback = []
        with open('feedback.txt') as infile:
            self.feedback = np.loadtxt(infile)
        with open('feat.txt') as infile:
            self.image_features = np.loadtxt(infile)
        with open('feedback_idx.txt') as infile:
            self.feedback_indices = np.loadtxt(infile)
        self.gp = None
        print("Haha")
    
    def Predict(self):
        print("Before cuda initialization")
        if not self.gp:
            self.gp = gp_cuda.GaussianProcessGPU(self.image_features,
                                                 self.feedback,
                                                 self.feedback_indices)
        print("After cuda initialization")

if __name__ == '__main__':
    GPSOM = GPSOM()
    GPSOM.Predict()