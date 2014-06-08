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

class GaussianProcessGPU:
    def __init__(self, img_features, img_shown_idx, block_size=(16, 16, 4)):
        self.block_size = block_size
        self.n_features = np.size(img_features, 1) # Assuming the n_features is divisible by block_size[2]
        # Add zero row to the beginning of feature matrix for zero padding in cuda operations
        self.img_features = np.asfarray(np.vstack(([np.zeros(self.n_features)], img_features)), dtype="float32")
        self.img_shown_idx = np.asarray(img_shown_idx, dtype="int32")


        cuda_source = open('../kernels.c', 'r')
        self.cuda_module = SourceModule(cuda_source.read())

        #self.n_total = np.size(self.img_features, 0) TODO: change these to use proper values after refactoring
        #self.n_shown = np.size(self.img_shown_idx, 0)
        self.n_total = 1024
        self.n_total_padded = self.round_to_blocksize(self.n_total)  # Pad to match block size
        self.n_shown = 4
        self.n_shown_padded = self.round_to_blocksize(self.n_shown)  # Pad to match block size

        self.n_predict = self.n_total - self.n_shown

        # Allocate GPU memory and copy data
        self.img_features_padded = self.img_features
        self.feat_gpu = drv.mem_alloc(self.img_features_padded.nbytes)
        drv.memcpy_htod(self.feat_gpu, self.img_features_padded)
        self.shown_idx_padded = np.asarray(
            np.concatenate((self.img_shown_idx, np.zeros(self.n_shown_padded - self.n_shown))),
            dtype="int32")
        self.shown_idx_gpu = drv.mem_alloc(self.shown_idx_padded.nbytes)
        drv.memcpy_htod(self.shown_idx_gpu, self.shown_idx_padded)

        # TODO: change to use proper values after refactoring
        self.predict = np.arange(0, self.n_total, dtype="int32")
        self.predict_gpu = drv.mem_alloc(self.predict.nbytes)
        drv.memcpy_htod(self.predict_gpu, self.predict)

    def round_to_blocksize(self, num):
        if num % self.block_size[0] != 0:
            return num + (self.block_size[0] - (num % self.block_size[0]))
        return num


    def gaussian_process(self, debug=False):
        K, K_noise = self.calc_K(output=True)
        if debug:
            print("K")
            print(K)

        feat_test = np.asfarray([self.img_features[i] for i in self.img_shown_idx])
        K_test = dist.cdist(feat_test, feat_test, 'cityblock') / self.n_features + np.diag(K_noise)
        if debug:
            print("K_test")
            print(np.allclose(K, K_test))

        Kx = self.calc_Kx()
        if debug:
            print("Kx")
            print(Kx)
            feat_test2 = np.asfarray([self.img_features[i] for i in self.predict], dtype="float32")
            Kx_test = dist.cdist(feat_test2, feat_test, 'cityblock') / self.n_features
            Kx_test = np.asfarray(Kx_test, dtype="float32")
            diff =  np.matrix(Kx_test) - np.matrix(Kx[:, :4])
            for line in diff.tolist():
                print(max(np.abs(line)))
            #TODO: something wrong with either test or the gpu calculation

    def moo(self):

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

        K_xK = np.zeros((n_total_img, n_shown_img), dtype = "float32")
        K_xK_gpu = drv.mem_alloc(K_xK.nbytes)

        CUBLAS_OP_N = 0
        alpha = 1.0
        beta = 0.0

        cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n_shown_img, n_total_img, n_shown_img, alpha, K_inv_gpu, n_shown_img, K_x_gpu, n_shown_img, beta, K_xK_gpu, n_shown_img)

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

        diag_K_xx = cumath.np.random.normal(1, 0.1, n_total_img)
        diag_K_xx = np.asfarray(diag_K_xx, dtype = "float32")
        diag_K_xx_gpu = drv.mem_alloc(diag_K_xx.nbytes)
        drv.memcpy_htod(diag_K_xx_gpu, diag_K_xx)

        #*******************************************************************************************************************************************************************************************************************************

        #print "diag_K_xx"
        #print diag_K_xx


        # diag_K_xKK_x_T
        #*******************************************************************************************************************************************************************************************************************************

        diag_K_xKK_x_T = np.zeros((1, n_total_img), dtype="float32")
        diag_K_xKK_x_T_gpu = drv.mem_alloc(diag_K_xKK_x_T.nbytes)
        drv.memcpy_htod(diag_K_xKK_x_T_gpu, diag_K_xKK_x_T)

        GRID_SIZE_x = (n_total_img + block_size - 1) / block_size
        GRID_SIZE_y = (1 + block_size - 1) / block_size

        func = mod.get_function("matMulDiag")
        func(K_xK_gpu, K_x_gpu, diag_K_xKK_x_T_gpu, np.int32(n_total_img), np.int32(n_shown_img), block = (block_size, block_size, 1), grid = (GRID_SIZE_x, GRID_SIZE_y, 1))

        drv.memcpy_dtoh(diag_K_xKK_x_T, diag_K_xKK_x_T_gpu)

        #print "diag_K_xKK_x_T"
        #print diag_K_xKK_x_T

        diag_K_xKK_x_T_test = np.diag(np.dot(K_xK, K_x.T))

        #print "diag_K_xKK_x_T_test"
        #print diag_K_xKK_x_T_test


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

    def calc_K(self, cleanup=False, output=False):
        """
        """
        grid_size_xy = (self.n_shown_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size_z = (self.n_features + self.block_size[2] - 1) / self.block_size[2]
        grid_size = (grid_size_xy, grid_size_xy, grid_size_z)
        K_padded = np.zeros((self.n_shown_padded, self.n_shown_padded), dtype="float32")
        # TODO: There is probably a better way to initialize zeroed memory on GPU for K_gpu

        # TODO: what's the point of doing this on GPU and transferring back and forth?
        K_noise = cumath.np.random.normal(1, 0.1, self.n_shown)  # Generate diagonal noise
        K_noise_padded = np.asfarray(np.concatenate((K_noise, np.zeros(self.n_shown_padded - self.n_shown))), dtype="float32")
        # Allocate memory and transfer
        K_noise_gpu = drv.mem_alloc(K_noise_padded.nbytes)
        self.K_gpu = drv.mem_alloc(K_padded.nbytes)
        drv.memcpy_htod(K_noise_gpu, K_noise_padded)
        drv.memcpy_htod(self.K_gpu, K_padded)

        func = self.cuda_module.get_function("generate__K__")
        func(self.K_gpu, self.shown_idx_gpu, self.feat_gpu, K_noise_gpu, np.int32(self.n_shown_padded), np.int32(self.n_features),
             block=self.block_size, grid=grid_size)

        if output:
            drv.memcpy_dtoh(K_padded, self.K_gpu)
        if cleanup:
            K_noise_gpu.free()
            self.K_gpu.free()
        if output:
            K = K_padded[:self.n_shown, :self.n_shown]
            print("K_padded")
            print(np.size(K_padded, 0), np.size(K_padded, 1))
            print(K_padded)
            return (K, K_noise)

    def calc_Kx(self, cleanup=False, output=False):
        grid_size_xy = (self.n_shown + self.block_size[0] - 1) / self.block_size[0]
        grid_size_z = (self.n_features + 4 - 1) / 4
        grid_size = (grid_size_xy, grid_size_xy, grid_size_z)

        K_x = np.zeros((self.n_total_padded, self.n_shown_padded), dtype="float32")
        self.K_x_gpu = drv.mem_alloc(K_x.nbytes)
        drv.memcpy_htod(self.K_x_gpu, K_x)


        func = self.cuda_module.get_function("generate__K_x__")
        func(self.K_x_gpu, self.shown_idx_gpu, self.predict_gpu, self.feat_gpu, np.int32(self.n_shown_padded),
             np.int32(self.n_predict), np.int32(self.n_features), block=self.block_size, grid=grid_size)

        drv.memcpy_dtoh(K_x, self.K_x_gpu)
        return(K_x)


if __name__ == "__main__":
    # Load image features
    feat = np.asfarray(np.load("../../data/cl25000.npy"), dtype="float32")
    gaussianProcess = GaussianProcessGPU(feat, np.arange(4, dtype="int32"))
    gaussianProcess.gaussian_process(debug=True)