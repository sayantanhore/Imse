# -*- coding: utf-8 -*-
"""
Created on Fri May 16 15:18:06 2014

@author: hore
"""

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import scikits.cuda.cublas as cublas
import numpy as np


# Kernel


mod = SourceModule("""


__global__ void generate_K_x_(float *K_x_gpu, float *shown_gpu, float *predict_gpu, float *feat_gpu, int BLOCK_SIZE, int PREDICTION_SIZE, int FEATURE_SIZE)
    {
        // Get co-ordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int x_counter = gridDim.x + blockDim.x;

        float distance = 0.0;

        if(x < BLOCK_SIZE && y < PREDICTION_SIZE)
        {
            int image_x = shown_gpu[x];
            int image_y = predict_gpu[y];

            for (int i = 0; i < FEATURE_SIZE; i++)
            {
                distance += abs(feat_gpu[image_x * FEATURE_SIZE + i]) + abs(feat_gpu[image_y * FEATURE_SIZE + i]);
            }

            K_x_gpu[y * x_counter + x] = distance;
        }
    }

__global__ void generate_K_(float *K_gpu, float *shown_gpu, float *feat_gpu, int SHOWN_SIZE, int FEATURE_SIZE)
    {
        // Get co-ordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int x_counter = gridDim.x + blockDim.x;

        float distance = 0.0;

        //if(x < SHOWN_SIZE && y < SHOWN_SIZE)
        //{
            int image_x = shown_gpu[x];
            int image_y = shown_gpu[y];

            for (int i = 0; i < FEATURE_SIZE; i++)
            {
                distance += abs(feat_gpu[image_x * FEATURE_SIZE + i]) + abs(feat_gpu[image_y * FEATURE_SIZE + i]);
            }

            K_gpu[y * x_counter + x] = distance / FEATURE_SIZE;
        //}
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


# Sigma

K = np.zeros((no_of_shown_images, no_of_shown_images), dtype = "float32")

K_gpu = drv.mem_alloc(K.nbytes)

func = mod.get_function("generate_K_")

GRID_SIZE_x = (no_of_shown_images + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (no_of_shown_images + BLOCK_SIZE - 1) / BLOCK_SIZE

func(K_gpu, shown_gpu, feat_gpu, np.int32(no_of_shown_images), np.int32(no_of_features), block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(K, K_gpu)

# Sigma_X

K_x = np.zeros((no_of_predictions, no_of_predictions), dtype = "float32")

K_x_gpu = drv.mem_alloc(K_x.nbytes)

func = mod.get_function("generate_K_x_")

GRID_SIZE_x = (no_of_predictions + BLOCK_SIZE - 1) / BLOCK_SIZE
GRID_SIZE_y = (no_of_shown_images + BLOCK_SIZE - 1) / BLOCK_SIZE

func(K_x_gpu, shown_gpu, predict_gpu, feat_gpu, block = (BLOCK_SIZE, BLOCK_SIZE, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

drv.memcpy_dtoh(K_x, K_x_gpu)

print "K::::"

print K
print K_x[22, 30]

# Free all memory

#feat_gpu.gpudata.free()

feat_gpu.free()

shown_gpu.free()

K_gpu.free()

h = cublas.cublasCreate()


print "Hello :: " + str(h)


cublas.cublasDestroy(h)