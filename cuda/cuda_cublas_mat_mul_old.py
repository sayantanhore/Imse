# -*- coding: utf-8 -*-
"""
Created on Fri May 16 15:18:06 2014

@author: hore
"""

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
import scikits.cuda.cublas as cublas
import numpy as np


# Kernel


mod = SourceModule("""

__global__ void generate__K__(float *K_gpu, float *shown_gpu, float *feat_gpu, float *K_noise_gpu, int SHOWN_SIZE, int FEATURE_SIZE)
    {
        // Get co-ordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int x_counter = gridDim.x + blockDim.x;

        float distance = 0.0;

        int image_x = shown_gpu[x];
        int image_y = shown_gpu[y];

        for (int i = 0; i < FEATURE_SIZE; i++)
        {
            distance += abs(feat_gpu[image_x * FEATURE_SIZE + i] - feat_gpu[image_y * FEATURE_SIZE + i]);
        }
        
        if(x == y)
        {
            K_gpu[y * x_counter + x] = distance / FEATURE_SIZE + K_noise_gpu[x];
        }
        else
        {
            K_gpu[y * x_counter + x] = distance / FEATURE_SIZE;            
        }

    }
__global__ void generate__K_x__(float *K_x_gpu, float *shown_gpu, float *predict_gpu, float *feat_gpu, int BLOCK_SIZE, int PREDICTION_SIZE, int FEATURE_SIZE)
    {
        // Get co-ordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int x_counter = gridDim.x + blockDim.x;

        float distance = 0.0;

        int image_x = shown_gpu[x];
        int image_y = predict_gpu[y];

        for (int i = 0; i < FEATURE_SIZE; i++)
        {
            distance += abs(feat_gpu[image_x * FEATURE_SIZE + i] - feat_gpu[image_y * FEATURE_SIZE + i]);
        }

        K_x_gpu[y * x_counter + x] = distance / FEATURE_SIZE;

    }
__global__ void generate__diag_K_xx__(float *diag_K_xx_gpu, float *K_xx_noise_gpu)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        diag_K_xx_gpu[x] = K_xx_noise_gpu[x];
    }
__global__ void generate__diag_K_xKK_x_T__(float *diag_K_xKK_x_T_gpu, float *K_xK_gpu, float *K_x, int K_xK_gpu_width)
    {
        float result = 0.0;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        
        for (int i = 0; i < K_xK_gpu_width; i++)
            {
                result += K_xK_gpu[x * K_xK_gpu_width + i] * K_x[x * K_xK_gpu_width + i];
            }
        diag_K_xKK_x_T_gpu[x] = result;
    }
__global__ void generate__variance__(float *variance_gpu, float *diag_K_xx_gpu, float *diag_K_xKK_x_T_gpu)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        variance_gpu[x] = sqrtf(diag_K_xx_gpu[x] - diag_K_xKK_x_T_gpu[x]);
    }
__global__ void generate__UCB__(float *ucb_gpu, float *mean_gpu, float *variance_gpu)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        ucb_gpu[x] = mean_gpu[x] + variance_gpu[x];
    }

""")


# Declarations

no_of_total_images = 25000

no_of_shown_images = 16

no_of_predictions = no_of_total_images - no_of_shown_images

no_of_features = 512

BLOCK_SIZE = 16



# Reading feature matrix in float32 format

feat = np.asfarray(np.load("/home/IMSE/data/Data/cl25000.npy"), dtype = "float32")

# Feature vectors
#feat_gpu = gpuarray.to_gpu(feat)

feat_gpu = drv.mem_alloc(feat.nbytes)

drv.memcpy_htod(feat_gpu, feat)


# Shown image number vector

shown = np.random.randint(0, 24999, no_of_shown_images)

shown_gpu = drv.mem_alloc(shown.nbytes)

drv.memcpy_htod(shown_gpu, shown)



# Prediction Set

predict = np.asarray([i for i in range(25000)])

predict_gpu = drv.mem_alloc(predict.nbytes)

drv.memcpy_htod(predict_gpu, predict)


# K
#*******************************************************************************************************************************************************************************************************************************

K = np.zeros((no_of_shown_images, no_of_shown_images), dtype = "float32")

K_gpu = drv.mem_alloc(K.nbytes)

K_noise = cumath.np.random.normal(1, 0.1, no_of_shown_images)

K_noise_gpu = drv.mem_alloc(K_noise.nbytes)

drv.memcpy_htod(K_noise_gpu, K_noise)



func = mod.get_function("generate__K__")

GRID_SIZE_x = (no_of_shown_images + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (no_of_shown_images + BLOCK_SIZE - 1) / BLOCK_SIZE

func(K_gpu, shown_gpu, feat_gpu, K_noise_gpu, np.int32(no_of_shown_images), np.int32(no_of_features), block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(K, K_gpu)

#*******************************************************************************************************************************************************************************************************************************

#print "K"
#print K

# K_x
#*******************************************************************************************************************************************************************************************************************************

K_x = np.zeros((no_of_total_images, no_of_shown_images), dtype = "float32")

K_x_gpu = drv.mem_alloc(K_x.nbytes)

func = mod.get_function("generate__K_x__")

GRID_SIZE_x = (no_of_shown_images + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (no_of_total_images + BLOCK_SIZE - 1) / BLOCK_SIZE

func(K_x_gpu, shown_gpu, predict_gpu, feat_gpu, block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(K_x, K_x_gpu)

#*******************************************************************************************************************************************************************************************************************************

#print "K_x"
#print K_x

# K_x.K
#*******************************************************************************************************************************************************************************************************************************

h = cublas.cublasCreate()

K_xK = np.zeros((no_of_total_images, no_of_shown_images), dtype = "float32")

K_xK_gpu = drv.mem_alloc(K_xK.nbytes)

CUBLAS_OP_N = 0
alpha = 1.0
beta = 0.0

cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, no_of_shown_images, no_of_total_images, no_of_shown_images, alpha, K_gpu, no_of_shown_images, K_x_gpu, no_of_shown_images, beta, K_xK_gpu, no_of_shown_images)

drv.memcpy_dtoh(K_xK, K_xK_gpu)

cublas.cublasDestroy(h)

#*******************************************************************************************************************************************************************************************************************************

#print "K_x.K"
#print K_xK

# diag_K_xx
#*******************************************************************************************************************************************************************************************************************************
'''
diag_K_xx = np.zeros((1, no_of_total_images), dtype = "float32")

diag_K_xx_gpu = drv.mem_alloc(diag_K_xx.nbytes)

func = mod.get_function("generate__diag_K_xx__")

GRID_SIZE_x = (no_of_total_images + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE

func(diag_K_xx_gpu, block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(diag_K_xx, diag_K_xx_gpu)
'''

diag_K_xx = cumath.np.random.normal(1, 0.1, no_of_total_images)
diag_K_xx_gpu = drv.mem_alloc(diag_K_xx.nbytes)

#*******************************************************************************************************************************************************************************************************************************

#print "diag_K_xx"
#print diag_K_xx

# diag_K_xKK_x_T
#*******************************************************************************************************************************************************************************************************************************

diag_K_xKK_x_T = np.zeros((1, no_of_total_images), dtype = "float32")

diag_K_xKK_x_T_gpu = drv.mem_alloc(diag_K_xKK_x_T.nbytes) 
func = mod.get_function("generate__diag_K_xKK_x_T__")

GRID_SIZE_x = (no_of_total_images + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE

func(diag_K_xKK_x_T_gpu, K_xK_gpu, K_x_gpu, np.int32(no_of_shown_images), block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(diag_K_xKK_x_T, diag_K_xKK_x_T_gpu)
#*******************************************************************************************************************************************************************************************************************************

#print "diag_K_xKK_x_T"
#print diag_K_xKK_x_T

# variance
#*******************************************************************************************************************************************************************************************************************************
variance = np.zeros((1, no_of_total_images), dtype = "float32")

variance_gpu = drv.mem_alloc(variance.nbytes)

func = mod.get_function("generate__variance__")

GRID_SIZE_x = (no_of_total_images + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE

func(variance_gpu, diag_K_xx_gpu, diag_K_xKK_x_T_gpu, block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(variance, variance_gpu)
#*******************************************************************************************************************************************************************************************************************************

#print "Variance"
#print variance

# Mean
#*******************************************************************************************************************************************************************************************************************************
feedback = np.array(np.random.random(no_of_shown_images))
feedback = np.array([feedback])

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

#print "Mean"
#print mean

# UCB
#*******************************************************************************************************************************************************************************************************************************
ucb = np.zeros((1, no_of_total_images), dtype = "float32")
ucb_gpu = drv.mem_alloc(ucb.nbytes)
func = mod.get_function("generate__UCB__")

GRID_SIZE_x = (no_of_total_images + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE

func(ucb_gpu, mean_gpu, variance_gpu, block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))
drv.memcpy_dtoh(ucb, ucb_gpu)
#*******************************************************************************************************************************************************************************************************************************

print "UCB"
print ucb


# Free all memory

#feat_gpu.gpudata.free()

feat_gpu.free()

shown_gpu.free()

K_gpu.free()

K_x_gpu.free()

K_xK_gpu.free()

diag_K_xx_gpu.free()

diag_K_xKK_x_T_gpu.free()

variance_gpu.free()

mean_gpu.free()

ucb_gpu.free()

