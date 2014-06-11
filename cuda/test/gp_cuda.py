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


def pad_vector(vector, n, n_pad, dtype=None):
    if dtype:
        return np.asarray(np.concatenate((vector, np.zeros(n_pad - n))), dtype=dtype)
    return np.asarray(np.concatenate((vector, np.zeros(n_pad - n))), dtype=vector.dtype)


def check_result(testname, A, B):
        print(testname + ' test')
        if not np.size(A) == np.size(B):
            print(testname + ' size check failed')
            print(testname + ': ' + str(np.size(A) + ' != ' + str(np.size(B))))
        if not np.allclose(A, B):
            print(testname + ' np.allclose test failed, np.isclose matrix:')
            print(np.isclose(A, B))
            print(testname + ' matrix:')
            print(A)
            print(testname + '_test matrix:')
            print(B)
        else:
            print(testname + ' test passed')


class GaussianProcessGPU:
    def __init__(self, img_features, img_shown_idx, block_size=(16, 16, 4)):
        self.float_type = np.float32
        self.int_type = np.int32
        self.block_size = block_size
        self.n_features = np.size(img_features, 1) # TODO: Assuming the n_features is divisible by block_size[2]


        cuda_source = open('../kernels.c', 'r')
        self.cuda_module = SourceModule(cuda_source.read())

        # Inialize variables
        # Pad everything to match block size
        # Add zero row to the beginning of feature matrix for zero padding in cuda operations TODO: is this necessary?
        self.img_features = np.asfarray(np.vstack(([np.zeros(self.n_features)], img_features)), dtype=self.float_type)
        self.n_total = np.size(self.img_features, 0)  # TODO: change these to use proper values after refactoring
        self.n_shown = np.size(img_shown_idx, 0)
#        self.n_total = self.int_type(220)
        self.n_total_padded = self.round_up_to_blocksize(self.n_total)  # Pad to match block size
#        self.n_shown = self.int_type(17)
        self.n_shown_padded = self.round_up_to_blocksize(self.n_shown)  # Pad to match block size
        self.n_predict = self.int_type(self.n_total - self.n_shown)
        self.n_predict_padded = self.round_up_to_blocksize(self.n_predict)
        self.shown_idx = np.asarray(img_shown_idx, dtype=self.int_type)
        self.shown_idx = pad_vector(self.shown_idx, self.n_shown, self.n_shown_padded, dtype=self.int_type)
        self.predict_idx = np.arange(0, self.n_predict, dtype=self.int_type)
        self.predict_idx = pad_vector(self.predict_idx, self.n_predict, self.n_predict_padded)
        self.K = np.zeros((self.n_shown_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_x = np.zeros((self.n_predict_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_xK = np.zeros((self.n_predict_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_noise = cumath.np.random.normal(1, 0.1, self.n_shown)  # Generate diagonal noise
        self.K_noise = pad_vector(self.K_noise, self.n_shown, self.n_shown_padded, dtype=self.float_type)
        self.K_inv = np.asfarray(self.K, dtype=self.float_type)
        self.diag_K_xx = cumath.np.random.normal(1, 0.1, self.n_total)
        self.diag_K_xx = pad_vector(self.diag_K_xx, self.n_total, self.n_total_padded, dtype=self.float_type)
        self.diag_K_xKK_x_T = np.zeros((1, self.n_predict_padded), dtype=self.float_type)
        self.variance = np.zeros((1, self.n_total_padded), dtype=self.float_type)
        self.feedback = np.array(np.random.random(self.n_shown))
        self.feedback = pad_vector(self.feedback, self.n_shown, self.n_shown_padded, dtype=self.float_type)
        self.mean = np.zeros((1, self.n_total_padded), dtype=self.float_type)
        self.ucb = np.zeros((1, self.n_total_padded), dtype=self.float_type)

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

        check_type(self.diag_K_xKK_x_T, self.float_type)
        self.diag_K_xKK_x_T_gpu = drv.mem_alloc(self.diag_K_xKK_x_T.nbytes)
        drv.memcpy_htod(self.diag_K_xKK_x_T_gpu, self.diag_K_xKK_x_T)

        check_type(self.variance, self.float_type)
        self.variance_gpu = drv.mem_alloc(self.variance.nbytes)

        check_type(self.feedback, self.float_type)
        self.feedback_gpu = drv.mem_alloc(self.feedback.nbytes)
        drv.memcpy_htod(self.feedback_gpu, self.feedback)

        check_type(self.mean, self.float_type)
        self.mean_gpu = drv.mem_alloc(self.mean.nbytes)

        check_type(self.ucb, self.float_type)
        self.ucb_gpu = drv.mem_alloc(self.ucb.nbytes)


    def round_up_to_blocksize(self, num):
        if num % self.block_size[0] != 0:
            return num + (self.block_size[0] - (num % self.block_size[0]))
        return self.int_type(num)


    def gaussian_process(self, debug=False):

        self.calc_K()
        if debug:
            K_test_features = np.asfarray([self.img_features[i] for i in self.shown_idx], dtype=self.float_type)
            K_test = dist.cdist(K_test_features, K_test_features, 'cityblock') / self.n_features + np.diag(self.K_noise)
            drv.memcpy_dtoh(self.K, self.K_gpu)
            check_result('K', self.K[:self.n_shown, :self.n_shown], K_test[:self.n_shown, :self.n_shown])

        self.invert_K()

        self.calc_K_x()
        if debug:
            drv.memcpy_dtoh(self.K_x, self.K_x_gpu)
            K_x_test = np.zeros((self.n_predict_padded, self.n_shown_padded), dtype=self.float_type)
            for i, idx1 in enumerate(self.predict_idx):
                for j, idx2 in enumerate(self.shown_idx):
                    vdist = distance(self.img_features[idx1], self.img_features[idx2]) / len(self.img_features[0])
                    K_x_test[i][j] = vdist
            check_result('K_x', self.K_x[:self.n_predict, :self.n_shown], K_x_test[:self.n_predict, :self.n_shown])

        self.calc_K_xK()
        if debug:
            drv.memcpy_dtoh(self.K_xK, self.K_xK_gpu)
            K_xK_test = (np.matrix(self.K_x) * np.matrix(self.K_inv))
            check_result('K_xK', self.K_xK[:self.n_predict, :self.n_shown], K_xK_test[:self.n_predict, :self.n_shown])

        self.calc_K_xKK_x_T()
        if debug:
            drv.memcpy_dtoh(self.diag_K_xKK_x_T, self.diag_K_xKK_x_T_gpu)
            K_xKK_x_T_test = np.diag(np.matrix(self.K_xK) * np.matrix(self.K_x).T)
            check_result("K_xKK_x_T test", self.diag_K_xKK_x_T, K_xKK_x_T_test)

        self.calc_variance()
        if debug:
            drv.memcpy_dtoh(self.variance, self.variance_gpu)
            variance_test = np.sqrt(np.abs(np.subtract(self.diag_K_xx[:self.n_predict], self.diag_K_xKK_x_T[:, :self.n_predict])))
            check_result('Variance', self.variance[:, :self.n_predict], variance_test[:, :self.n_predict])

        self.calc_mean()
        if debug:
            drv.memcpy_dtoh(self.mean, self.mean_gpu)
            mean_test = np.dot(self.K_xK, self.feedback)  # This is 1D, self.mean is 2D, so slicing for test differs
            check_result('Mean', self.mean[:, :self.n_predict], mean_test[:self.n_predict])

        self.calc_UCB()
        if debug:
            drv.memcpy_dtoh(self.ucb, self.ucb_gpu)
            ucb_test = np.add(self.mean, self.variance)  # The array shapes differ a bit, so slicing is different
            check_result('UCB', self.ucb[:self.n_predict, :], ucb_test[:self.n_predict])

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
        tmp = cumath.np.linalg.inv(K[:self.n_shown, :self.n_shown])
        self.K_inv[:tmp.shape[0], :tmp.shape[1]] = tmp
        drv.memcpy_htod(self.K_inv_gpu, self.K_inv)

    def calc_K_x(self):
        grid_size_x = (self.n_shown_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size_y = (self.n_predict_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size_z = (self.n_features + self.block_size[2] - 1) / self.block_size[2]
        grid_size = (grid_size_x, grid_size_y, grid_size_z)

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

    def calc_variance(self):
        grid_size_x = (self.n_total_padded + self.block_size[0]- 1) / self.block_size[0]
        grid_size_y = (1 + self.block_size[0] - 1) / self.block_size[0]
        grid_size = (grid_size_x, grid_size_y, 1)
        cuda_func = self.cuda_module.get_function("generate__variance__")
        cuda_func(self.variance_gpu, self.diag_K_xx_gpu, self.diag_K_xKK_x_T_gpu, np.int32(self.n_total),
                  block=(self.block_size[0], self.block_size[0], 1), grid=grid_size)

    def calc_mean(self):
        h = cublas.cublasCreate()
        CUBLAS_OP_N = 0
        alpha = 1.0
        beta = 0.0
        cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, self.n_total_padded, self.n_shown_padded, alpha,
                           self.feedback_gpu, 1, self.K_xK_gpu, self.n_shown_padded, beta, self.mean_gpu, 1)
        cublas.cublasDestroy(h)

    def calc_UCB(self):
        grid_size_x = (self.n_total_padded + self.block_size[0]- 1) / self.block_size[0]
        grid_size = (grid_size_x, 1, 1)
        cuda_func = self.cuda_module.get_function("generate__UCB__")
        cuda_func(self.ucb_gpu, self.mean_gpu, self.variance_gpu, block=(self.block_size[0], 1, 1), grid=grid_size)


if __name__ == "__main__":
    # Load image features
    feat = np.asfarray(np.load("../../data/cl25000.npy"), dtype="float32")
    gaussianProcess = GaussianProcessGPU(feat, np.arange(33, dtype="int32"))
    gaussianProcess.gaussian_process(debug=False)
