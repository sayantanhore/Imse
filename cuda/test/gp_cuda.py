# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:30:41 2014

@author: hore
"""

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
import scikits.cuda.cublas as cublas
import numpy as np
import scipy.spatial.distance as dist

# Load Kernel

cuda_source = open('kernels.c', 'r')
mod = SourceModule(cuda_source.read())

# Declarations

no_of_total_images = 25000
no_of_shown_images = 16
no_of_predictions = no_of_total_images - no_of_shown_images
no_of_features = 512
block_size = 16


# Reading feature matrix in float32 format

feat = np.asfarray(np.load("/home/IMSE/data/Data/cl25000.npy"), dtype = "float32")
feat_gpu = drv.mem_alloc(feat.nbytes)
drv.memcpy_htod(feat_gpu, feat)


# Reading shown vector

#shown = np.random.randint(0, 24999, no_of_shown_images)
shown = np.arange(0, no_of_shown_images, dtype="int32")
shown = np.asarray(shown, dtype = "int32")
shown_gpu = drv.mem_alloc(shown.nbytes)
drv.memcpy_htod(shown_gpu, shown)


# Prediction Set

predict = np.arange(0, no_of_total_images, dtype="int32")
predict_gpu = drv.mem_alloc(predict.nbytes)
drv.memcpy_htod(predict_gpu, predict)


# K
#*******************************************************************************************************************************************************************************************************************************

K = np.zeros((no_of_shown_images, no_of_shown_images), dtype = "float32")
K_gpu = drv.mem_alloc(K.nbytes)
drv.memcpy_htod(K_gpu, K)

K_noise = cumath.np.random.normal(1, 0.1, no_of_shown_images)
K_noise = np.asfarray(K_noise, dtype = "float32")
K_noise_gpu = drv.mem_alloc(K_noise.nbytes)
drv.memcpy_htod(K_noise_gpu, K_noise)


GRID_SIZE_x = (no_of_shown_images + block_size - 1) / block_size
GRID_SIZE_y = (no_of_shown_images + block_size - 1) / block_size
GRID_SIZE_z = (no_of_features + 4 - 1) / 4

func = mod.get_function("generate__K__")
func(K_gpu, shown_gpu, feat_gpu, K_noise_gpu, np.int32(no_of_shown_images), np.int32(no_of_features), block = (block_size, block_size, 4), grid = (GRID_SIZE_x, GRID_SIZE_y, GRID_SIZE_z))

drv.memcpy_dtoh(K, K_gpu)
#*******************************************************************************************************************************************************************************************************************************

#print "K"
#print K

feat_test = feat[shown, :]
K_test = dist.cdist(feat_test, feat_test, 'cityblock') / no_of_features + np.diag(K_noise)

#print "K_test"
#print K_test



# K_x
#*******************************************************************************************************************************************************************************************************************************

K_x = np.zeros((no_of_total_images, no_of_shown_images), dtype="float32")
K_x_gpu = drv.mem_alloc(K_x.nbytes)
drv.memcpy_htod(K_x_gpu, K_x)

GRID_SIZE_x = (no_of_shown_images + block_size - 1) / block_size
GRID_SIZE_y = (no_of_total_images + block_size - 1) / block_size
GRID_SIZE_z = (no_of_features + 4 - 1) / 4

func = mod.get_function("generate__K_x__")
func(K_x_gpu, shown_gpu, predict_gpu, feat_gpu, np.int32(block_size), np.int32(no_of_total_images), np.int32(no_of_features), block = (block_size, block_size, 4), grid = (GRID_SIZE_x, GRID_SIZE_y, GRID_SIZE_z))

drv.memcpy_dtoh(K_x, K_x_gpu)
#*******************************************************************************************************************************************************************************************************************************

#print "K_x"
#print K_x

feat_test = feat[shown, :]
K_test = dist.cdist(feat, feat_test, 'cityblock') / no_of_features

#print "K_x"
#print K_x

# K_inv
#*******************************************************************************************************************************************************************************************************************************

K_inv = cumath.np.linalg.inv(K)
K_inv = np.asfarray(K_inv, dtype = "float32")
K_inv_gpu = drv.mem_alloc(K_inv.nbytes)
drv.memcpy_htod(K_inv_gpu, K_inv)
#*******************************************************************************************************************************************************************************************************************************

#print "K_inv"
#print K_inv

K_inv_test = np.linalg.inv(K)

#print "K_inv"
#print K_inv_test

#print np.all(K_inv == K_inv_test)


# K_x.K
#*******************************************************************************************************************************************************************************************************************************

h = cublas.cublasCreate()

K_xK = np.zeros((no_of_total_images, no_of_shown_images), dtype = "float32")
K_xK_gpu = drv.mem_alloc(K_xK.nbytes)

CUBLAS_OP_N = 0
alpha = 1.0
beta = 0.0

cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, no_of_shown_images, no_of_total_images, no_of_shown_images, alpha, K_inv_gpu, no_of_shown_images, K_x_gpu, no_of_shown_images, beta, K_xK_gpu, no_of_shown_images)

drv.memcpy_dtoh(K_xK, K_xK_gpu)

cublas.cublasDestroy(h)
#*******************************************************************************************************************************************************************************************************************************


#print "K_xK"
#print K_xK


K_xK_test = np.dot(K_x, K)

#print "K_xK"
#print K_xK


# diag_K_xx
#*******************************************************************************************************************************************************************************************************************************

diag_K_xx = cumath.np.random.normal(1, 0.1, no_of_total_images)
diag_K_xx = np.asfarray(diag_K_xx, dtype = "float32")
diag_K_xx_gpu = drv.mem_alloc(diag_K_xx.nbytes)
drv.memcpy_htod(diag_K_xx_gpu, diag_K_xx)

#*******************************************************************************************************************************************************************************************************************************

#print "diag_K_xx"
#print diag_K_xx


# diag_K_xKK_x_T
#*******************************************************************************************************************************************************************************************************************************

diag_K_xKK_x_T = np.zeros((1, no_of_total_images), dtype="float32")
diag_K_xKK_x_T_gpu = drv.mem_alloc(diag_K_xKK_x_T.nbytes)
drv.memcpy_htod(diag_K_xKK_x_T_gpu, diag_K_xKK_x_T)

GRID_SIZE_x = (no_of_total_images + block_size - 1) / block_size
GRID_SIZE_y = (1 + block_size - 1) / block_size

func = mod.get_function("matMulDiag")
func(K_xK_gpu, K_x_gpu, diag_K_xKK_x_T_gpu, np.int32(no_of_total_images), np.int32(no_of_shown_images), block = (block_size, block_size, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(diag_K_xKK_x_T, diag_K_xKK_x_T_gpu)

#print "diag_K_xKK_x_T"
#print diag_K_xKK_x_T

diag_K_xKK_x_T_test = np.diag(np.dot(K_xK, K_x.T))

#print "diag_K_xKK_x_T_test"
#print diag_K_xKK_x_T_test


# variance
#*******************************************************************************************************************************************************************************************************************************
variance = np.zeros((1, no_of_total_images), dtype = "float32")
variance_gpu = drv.mem_alloc(variance.nbytes)
func = mod.get_function("generate__variance__")

GRID_SIZE_x = (no_of_total_images + block_size - 1) / block_size
GRID_SIZE_y = (1 + block_size - 1) / block_size

func(variance_gpu, diag_K_xx_gpu, diag_K_xKK_x_T_gpu, np.int32(no_of_total_images), block = (block_size, block_size, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(variance, variance_gpu)
#*******************************************************************************************************************************************************************************************************************************

print "Variance"
print variance

variance_test = np.sqrt(np.abs(np.subtract(diag_K_xx, diag_K_xKK_x_T)))

print "Variance_test"
print variance_test


# Mean
#*******************************************************************************************************************************************************************************************************************************
feedback = np.array(np.random.random(no_of_shown_images))
feedback = np.array([feedback])
feedback = np.asfarray(feedback, dtype = "float32")

feedback_gpu = drv.mem_alloc(feedback.nbytes)
drv.memcpy_htod(feedback_gpu, feedback)

h = cublas.cublasCreate()

mean = np.zeros((1, no_of_total_images), dtype = "float32")
mean_gpu = drv.mem_alloc(mean.nbytes)

CUBLAS_OP_N = 0
alpha = 1.0
beta = 0.0

cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, no_of_total_images, no_of_shown_images, alpha, feedback_gpu, 1, K_xK_gpu, no_of_shown_images, beta, mean_gpu, 1)
drv.memcpy_dtoh(mean, mean_gpu)

cublas.cublasDestroy(h)
#*******************************************************************************************************************************************************************************************************************************


print "mean"
print mean

mean_test = np.dot(K_xK, feedback.T)

print "mean_test"
print mean_test.T

# UCB
#*******************************************************************************************************************************************************************************************************************************
ucb = np.zeros((1, no_of_total_images), dtype = "float32")
ucb_gpu = drv.mem_alloc(ucb.nbytes)

GRID_SIZE_x = (no_of_total_images + block_size - 1) / block_size
GRID_SIZE_y = (1 + block_size - 1) / block_size

func = mod.get_function("generate__UCB__")
func(ucb_gpu, mean_gpu, variance_gpu, block = (block_size, block_size, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))
drv.memcpy_dtoh(ucb, ucb_gpu)

#*******************************************************************************************************************************************************************************************************************************

print "UCB"
print ucb

ucb_test = np.add(mean, variance)

print "ucb_test"
print ucb_test