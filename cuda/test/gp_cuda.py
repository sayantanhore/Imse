#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:30:41 2014

@author: sayantan, lasse
"""

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
import scikits.cuda.cublas as cublas
import numpy as np
import scipy.spatial.distance as dist

def distance(vector1, vector2, metric="manhattan"):
    vdist = 0
    if metric == "manhattan":
        for i in range(len(vector1)):
            vdist += abs(vector1[i] - vector2[i])
    else:
        raise ValueError('Invalid parameter: '+str(metric))
    return  vdist

def check_type(variable, dtype):
    if variable.dtype != dtype:
        raise TypeError('Invalid K_x dtype: ' + variable.dtype + ', expected ' + dtype)

class GaussianProcessGPU:
    def __init__(self, img_features, img_shown_idx, block_size=(16, 16, 4)):
        self.float_type = np.float32
        self.int_type = np.int32
        self.block_size = block_size
        self.n_features = np.size(img_features, 1) # TODO: Assuming the n_features is divisible by block_size[2]


        cuda_source = open('../kernels.c', 'r')
        self.cuda_module = SourceModule(cuda_source.read())

        # Pad everything to match block size
        # Add zero row to the beginning of feature matrix for zero padding in cuda operations
        self.img_features = np.asfarray(np.vstack(([np.zeros(self.n_features)], img_features)), dtype=self.float_type)
        self.shown_idx = np.asarray(img_shown_idx, dtype=self.int_type)
        #self.n_total = np.size(self.img_features, 0) TODO: change these to use proper values after refactoring
        #self.n_shown = np.size(self.img_shown_idx, 0)
        self.n_total = self.int_type(16)
        self.n_total_padded = self.round_up_to_blocksize(self.n_total)  # Pad to match block size
        self.n_shown = self.int_type(4)
        self.n_shown_padded = self.round_up_to_blocksize(self.n_shown)  # Pad to match block size
        self.n_predict = self.int_type(self.n_total - self.n_shown)
        self.n_predict_padded = self.round_up_to_blocksize(self.n_predict)
        self.shown_idx = np.asarray(
            np.concatenate((self.shown_idx, np.zeros(self.n_shown_padded - self.n_shown))),
            dtype=self.int_type)
        self.predict_idx = np.arange(0, self.n_predict, dtype=self.int_type)
        self.predict_idx = np.asarray(
            np.concatenate((self.predict_idx, np.zeros(self.n_predict_padded - self.n_predict))),
            dtype=self.int_type)
        self.K = np.zeros((self.n_shown_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_x = np.zeros((self.n_shown_padded, self.n_predict_padded), dtype=self.float_type)
        self.K_xK = np.zeros((self.n_predict_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_noise = cumath.np.random.normal(1, 0.1, self.n_shown)  # Generate diagonal noise
        self.K_noise = np.asfarray(
            np.concatenate((self.K_noise, np.zeros(self.n_shown_padded - self.n_shown))),
            dtype=self.float_type)
        self.K_inv = np.asfarray(self.K, dtype="float32")
        self.diag_K_xx = cumath.np.random.normal(1, 0.1, self.n_total)
        self.diag_K_xx = np.asfarray(
            np.concatenate((self.diag_K_xx, np.zeros(self.n_total_padded - self.n_total))),
            dtype=self.float_type)

        self.diag_K_xKK_x_T = np.zeros((1, self.n_total_padded), dtype="float32")




        # Allocate GPU memory and copy data, check datatype before each allocation
        # TODO: add dimension checking
        check_type(self.img_features, self.float_type)
        self.feat_gpu = drv.mem_alloc(self.img_features.nbytes)
        drv.memcpy_htod(self.feat_gpu, self.img_features)

        check_type(self.shown_idx, self.int_type)
        self.shown_idx_gpu = drv.mem_alloc(self.shown_idx.nbytes)
        drv.memcpy_htod(self.shown_idx_gpu, self.shown_idx)

        check_type(self.K, self.float_type)
        self.K_gpu = drv.mem_alloc(self.K.nbytes)
        drv.memcpy_htod(self.K_gpu, self.K)

        check_type(self.K_inv, self.float_type)
        self.K_inv_gpu = drv.mem_alloc(self.K_inv.nbytes)

        check_type(self.K_noise, self.float_type)
        self.K_noise_gpu = drv.mem_alloc(self.K_noise.nbytes)
        drv.memcpy_htod(self.K_noise_gpu, self.K_noise)

        check_type(self.K_x, self.float_type)
        self.K_x_gpu = drv.mem_alloc(self.K_x.nbytes)
        drv.memcpy_htod(self.K_x_gpu, self.K_x)

        check_type(self.predict_idx, self.int_type)
        self.predict_idx_gpu = drv.mem_alloc(self.predict_idx.nbytes)
        drv.memcpy_htod(self.predict_idx_gpu, self.predict_idx)

        check_type(self.K_xK, self.float_type)
        self.K_xK_gpu = drv.mem_alloc(self.K_xK.nbytes)

        check_type(self.diag_K_xx, self.float_type)
        self.diag_K_xx_gpu = drv.mem_alloc(self.diag_K_xx.nbytes)
        drv.memcpy_htod(self.diag_K_xx_gpu, self.diag_K_xx)

        self.diag_K_xKK_x_T_gpu = drv.mem_alloc(self.diag_K_xKK_x_T.nbytes)
        drv.memcpy_htod(self.diag_K_xKK_x_T_gpu, self.diag_K_xKK_x_T)



    def round_up_to_blocksize(self, num):
        if num % self.block_size[0] != 0:
            return num + (self.block_size[0] - (num % self.block_size[0]))
        return self.int_type(num)

    def gaussian_process(self, debug=False):
        K_test_features = np.asfarray([self.img_features[i] for i in self.shown_idx], dtype=self.float_type)
        K_test = dist.cdist(K_test_features, K_test_features, 'cityblock') / self.n_features + np.diag(self.K_noise)

        self.calc_K()
        if debug:
            drv.memcpy_dtoh(self.K, self.K_gpu)
            print("K")
            print(self.K)
            print(np.matrix(K_test) - np.matrix(self.K))
            print(np.allclose(self.K, K_test))

        self.invert_K()

        K_x_test_features = np.asfarray(self.img_features, dtype=self.float_type)
        self.calc_K_x()
        if debug:
            drv.memcpy_dtoh(self.K_x, self.K_x_gpu)
            print("Kx")
            print(self.K_x)
            print(self.predict_idx)
            K_x_test = np.zeros((self.n_total_padded, self.n_shown_padded), dtype="float32")
            for i, idx1 in enumerate(self.predict_idx):
                for j, idx2 in enumerate(self.shown_idx):
                    vdist = distance(self.img_features[idx1], self.img_features[idx2]) / len(self.img_features[0])
                    K_x_test[i][j] = vdist
            print("K_x_test")
            print(K_x_test)
            diff = np.isclose(np.matrix(K_x_test[:, :5]), np.matrix(self.K_x[:, :5]))
            print(diff)
            #for line in Kx.tolist():
            #    print(line)
            #TODO: something wrong with the test

        self.calc_K_xK()
        if debug:
            drv.memcpy_dtoh(self.K_xK, self.K_xK_gpu)
            print("K_xK test")
            K_xK_test = (np.matrix(self.K_x) * np.matrix(self.K_inv))
            print(np.isclose(self.K_xK, K_xK_test))

        self.calc_K_xKK_x_T()
        if debug:
            drv.memcpy_dtoh(self.diag_K_xKK_x_T, self.diag_K_xKK_x_T_gpu)
            print("K_xKK_x_T test")
            print(np.isclose(self.diag_K_xKK_x_T, np.diag(np.matrix(self.K_xK) * np.matrix(self.K_x).T)))


    def moo(self):

        # variance
        #*******************************************************************************************************************************************************************************************************************************
        variance = np.zeros((1, n_total_img), dtype = "float32")
        variance_gpu = drv.mem_alloc(variance.nbytes)
        func = mod.get_function("generate__variance__")

        GRID_SIZE_x = (n_total_img + block_size - 1) / block_size
        GRID_SIZE_y = (1 + block_size - 1) / block_size

        func(variance_gpu, diag_K_xx_gpu, diag_K_xKK_x_T_gpu, np.int32(n_total_img), block = (block_size, block_size, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

        drv.memcpy_dtoh(variance, variance_gpu)
        #*******************************************************************************************************************************************************************************************************************************

        print "Variance"
        print variance

        variance_test = np.sqrt(np.abs(np.subtract(diag_K_xx, diag_K_xKK_x_T)))

        print "Variance_test"
        print variance_test


        # Mean
        #*******************************************************************************************************************************************************************************************************************************
        feedback = np.array(np.random.random(n_shown_img))
        feedback = np.array([feedback])
        feedback = np.asfarray(feedback, dtype = "float32")

        feedback_gpu = drv.mem_alloc(feedback.nbytes)
        drv.memcpy_htod(feedback_gpu, feedback)

        h = cublas.cublasCreate()

        mean = np.zeros((1, n_total_img), dtype = "float32")
        mean_gpu = drv.mem_alloc(mean.nbytes)

        CUBLAS_OP_N = 0
        alpha = 1.0
        beta = 0.0

        cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, n_total_img, n_shown_img, alpha, feedback_gpu, 1, K_xK_gpu, n_shown_img, beta, mean_gpu, 1)
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
        ucb = np.zeros((1, n_total_img), dtype = "float32")
        ucb_gpu = drv.mem_alloc(ucb.nbytes)

        GRID_SIZE_x = (n_total_img + block_size - 1) / block_size
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

    def calc_K(self):
        """
        """
        grid_size_xy = (self.n_shown_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size_z = (self.n_features + self.block_size[2] - 1) / self.block_size[2]
        grid_size = (grid_size_xy, grid_size_xy, grid_size_z)


        cuda_func = self.cuda_module.get_function("generate__K__")
        cuda_func(self.K_gpu, self.shown_idx_gpu, self.feat_gpu, self.K_noise_gpu, np.int32(self.n_shown_padded),
                  np.int32(self.n_features), block=self.block_size, grid=grid_size)

    def invert_K(self):
        K = np.zeros((self.n_shown_padded, self.n_shown_padded), dtype=self.float_type)
        drv.memcpy_dtoh(K, self.K_gpu)
        print(K)
        tmp = cumath.np.linalg.inv(K[:self.n_shown, :self.n_shown])
        self.K_inv[:tmp.shape[0], :tmp.shape[1]] = tmp
        print("K_inv")
        print(self.K_inv)
        drv.memcpy_htod(self.K_inv_gpu, self.K_inv)

    def calc_K_x(self):
        grid_size_xy = (self.n_shown_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size_z = (self.n_features + self.block_size[2] - 1) / self.block_size[2]
        grid_size = (grid_size_xy, grid_size_xy, grid_size_z)

        cuda_func = self.cuda_module.get_function("generate__K_x__")
        cuda_func(self.K_x_gpu, self.shown_idx_gpu, self.predict_idx_gpu, self.feat_gpu, np.int32(self.n_shown_padded),
             np.int32(self.n_predict_padded), np.int32(self.n_features), block=self.block_size, grid=grid_size)

    def calc_K_xK(self):
        h = cublas.cublasCreate()
        CUBLAS_OP_N = 0
        alpha = 1.0
        beta = 0.0
        cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, self.n_shown_padded, self.n_predict_padded, self.n_shown_padded,
                           alpha, self.K_inv_gpu, self.n_shown_padded, self.K_x_gpu, self.n_shown_padded, beta, self.K_xK_gpu,
                           self.n_shown_padded)
        cublas.cublasDestroy(h)

    def calc_K_xKK_x_T(self):
        grid_size_x = (self.n_total_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size_y = (1 + self.block_size[0] - 1) / self.block_size[0]
        grid_size = (grid_size_x, grid_size_y, 1)

        cuda_func = self.cuda_module.get_function("matMulDiag")
        cuda_func(self.K_xK_gpu, self.K_x_gpu, self.diag_K_xKK_x_T_gpu, np.int32(self.n_total_padded),
                  np.int32(self.n_shown_padded), block=(self.block_size[0], self.block_size[0], 1), grid = grid_size)


if __name__ == "__main__":
    # Load image features
    feat = np.asfarray(np.load("../../data/cl25000.npy"), dtype="float32")
    gaussianProcess = GaussianProcessGPU(feat, np.arange(4, dtype="int32"))
    gaussianProcess.gaussian_process(debug=True)
