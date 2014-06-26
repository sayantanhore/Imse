#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:30:41 2014

@author: sayantan, lasse
"""

import pycuda.driver as drv
import pycuda.tools
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
    return vdist


def check_type(variable, dtype):
    if variable.dtype != dtype:
        raise TypeError('Invalid variable dtype: ' + str(variable.dtype) + ', expected ' + str(dtype))


def check_dimensions(array, dims):  # TODO: implement proper checking in code
    if np.shape(array) != dims:
        raise ValueError('Invalid array shape: ' + str(np.shape(array)) + ', expected ' + str(dims))


def pad_vector(vector, n, n_pad, dtype=None):
    if dtype:
        return np.asarray(np.concatenate((vector, np.zeros(n_pad - n))), dtype=dtype)
    return np.asarray(np.concatenate((vector, np.zeros(n_pad - n))), dtype=vector.dtype)


def check_result(testname, A, B):
        print(testname + ' test')
        if not np.size(A) == np.size(B):
            print(testname + ' size check failed')
            print(testname + ': ' + str(np.size(A)) + ' != ' + str(np.size(B)))
        if not np.allclose(A, B):
            print(testname + ' np.allclose test failed, np.isclose matrix:')
            print(np.isclose(A, B))
            print(testname + ' matrix:')
            print(A)
            print(testname + '_test matrix:')
            print(B)
        else:
            print(testname + ' test passed')


def allocate_array(array, gpu_array=None, zeros=True):
    """
    Allocate space for array in GPU memory.
    Parameters:
        array is the array to allocate
    Returns:
        reference to the allocated gpu_array
    """
    if gpu_array:
        gpu_array.free()
    gpu_array = drv.mem_alloc(array.nbytes)
    if zeros:
        drv.memset_d32(gpu_array, 0, array.size)
    return gpu_array


class GaussianProcessGPU:
    """
    Gaussian process class, which uses GPU for the distance and matrix calculations. This implementation is not
    thoroughly tested, for example large imagesets will probably cause it to fail.

    Constructing GaussianProcessGPU object:
    GaussianProcessGPU(img_features, feedback, img_shown_idx, block_size=(16, 16, 4))

    Parameters:
        img_features is a 2D array, each row containing float features for one image.
        block_size is the block size used in the GPU kernel calls. Don't change unless you know what you're doing.
        float_type defines which numpy float type is used (tested with float32, may or may not work with float64)
        int_type defines which numpy int type is used (tested with int32, may or may not work with int64)
        kernel_file is the file from which GPU kernels are read.
    """
    def __init__(self, img_features, block_size=(16, 16, 4), float_type=np.float32,
                 int_type=np.int32, kernel_file='../kernels.c'):
        import pycuda.autoinit
        np.set_printoptions(linewidth=500)

        self.float_type = float_type
        self.int_type = int_type
        self.block_size = block_size
        self.n_features = np.size(img_features, 1) # TODO: Cuda calls are assuming the n_features is divisible by block_size[2], fix

        cuda_source = open(kernel_file, 'r')
        self.cuda_module = SourceModule(cuda_source.read())

        # Inialize variables
        # Pad everything to match block size
        # Add zero row to the beginning of feature matrix for zero padding in cuda operations TODO: is this necessary?
        self.img_features = np.asfarray(np.vstack(([np.zeros(self.n_features)], img_features)), dtype=self.float_type)
        self.n_total = np.size(self.img_features, 0)
        self.n_total_padded = self.round_up_to_blocksize(self.n_total)  # Pad to match block size

        check_type(self.img_features, self.float_type)
        self.feat_gpu = drv.mem_alloc(self.img_features.nbytes)
        drv.memcpy_htod(self.feat_gpu, self.img_features)

        self.n_shown = None
        self.n_shown_padded = None
        self.n_predict = None
        self.n_predict_padded = None
        self.shown_idx = None
        self.shown_idx_gpu = None
        self.predict_idx = None
        self.K = None
        self.K_x = None
        self.K_xK = None
        self.K_noise = None
        self.K_noise = None
        self.K_inv = None
        self.diag_K_xx = None
        self.diag_K_xx = None
        self.diag_K_xKK_x_T = None
        self.variance = None
        self.feedback = None
        self.feedback = None
        self.mean = None
        self.ucb = None
        self.shown_idx_gpu = None
        self.K_gpu = None
        self.K_inv_gpu = None
        self.K_noise_gpu = None
        self.K_x_gpu = None
        self.predict_idx_gpu = None
        self.K_xK_gpu = None
        self.diag_K_xx_gpu = None
        self.diag_K_xKK_x_T_gpu = None
        self.variance_gpu = None
        self.feedback_gpu = None
        self.mean_gpu = None
        self.ucb_gpu = None

    def set_feedback(self, feedback, feedback_indices):
        """
        :param feedback: is a vector containing at least one initial observation as a float
        :param feedback_indices: is a vector containing the indexes of the images for which feedback was given in the
            feedback vector
        :return:
        """
        self.n_shown = np.size(feedback_indices, 0)
        self.n_shown_padded = self.round_up_to_blocksize(self.n_shown)  # Pad to match block size
        self.n_predict = self.int_type(self.n_total - self.n_shown)
        self.n_predict_padded = self.round_up_to_blocksize(self.n_predict)
        self.shown_idx = np.asarray(feedback_indices, dtype=self.int_type)
        self.shown_idx = pad_vector(self.shown_idx, self.n_shown, self.n_shown_padded, dtype=self.int_type)
        self.predict_idx = np.arange(0, int(self.n_predict), dtype=self.int_type)
        self.predict_idx = pad_vector(self.predict_idx, self.n_predict, self.n_predict_padded)
        self.K = np.zeros((self.n_shown_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_x = np.zeros((self.n_predict_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_xK = np.zeros((self.n_predict_padded, self.n_shown_padded), dtype=self.float_type)
        self.K_noise = cumath.np.random.normal(1, 0.1, self.n_shown)  # Generate diagonal noise
        self.K_noise = pad_vector(self.K_noise, self.n_shown, self.n_shown_padded, dtype=self.float_type)
        self.K_inv = np.asfarray(self.K, dtype=self.float_type)
        self.diag_K_xx = cumath.np.random.normal(1, 0.1, self.n_predict)
        self.diag_K_xx = pad_vector(self.diag_K_xx, self.n_predict, self.n_predict_padded, dtype=self.float_type)
        self.diag_K_xKK_x_T = np.zeros((1, self.n_predict_padded), dtype=self.float_type)
        self.variance = np.zeros((1, self.n_predict_padded), dtype=self.float_type)
        self.feedback = np.array(feedback)
        self.feedback = pad_vector(self.feedback, self.n_shown, self.n_shown_padded, dtype=self.float_type)
        self.mean = np.zeros((1, self.n_predict_padded), dtype=self.float_type)
        self.ucb = np.zeros((1, self.n_predict_padded), dtype=self.float_type)

        # Allocate GPU memory and copy data, check datatype before each allocation
        # TODO: add dimension checking

        check_type(self.shown_idx, self.int_type)
        self.shown_idx_gpu = allocate_array(self.shown_idx, self.shown_idx_gpu, zeros=False)
        drv.memcpy_htod(self.shown_idx_gpu, self.shown_idx)

        check_type(self.K, self.float_type)
        self.K_gpu = allocate_array(self.K, self.K_gpu)
        #drv.memcpy_htod(self.K_gpu, self.K)

        check_type(self.K_inv, self.float_type)
        self.K_inv_gpu = allocate_array(self.K_inv, self.K_inv_gpu)

        check_type(self.K_noise, self.float_type)
        self.K_noise_gpu = allocate_array(self.K_noise, self.K_noise_gpu, zeros=False)
        drv.memcpy_htod(self.K_noise_gpu, self.K_noise)

        check_type(self.K_x, self.float_type)
        self.K_x_gpu = allocate_array(self.K_x, self.K_x_gpu)
        #drv.memcpy_htod(self.K_x_gpu, self.K_x)

        check_type(self.predict_idx, self.int_type)
        self.predict_idx_gpu = allocate_array(self.predict_idx, self.predict_idx_gpu)
        drv.memcpy_htod(self.predict_idx_gpu, self.predict_idx)

        check_type(self.K_xK, self.float_type)
        self.K_xK_gpu = allocate_array(self.K_xK, self.K_xK_gpu)

        check_type(self.diag_K_xx, self.float_type)
        self.diag_K_xx_gpu = allocate_array(self.diag_K_xx, self.diag_K_xx_gpu, zeros=False)
        drv.memcpy_htod(self.diag_K_xx_gpu, self.diag_K_xx)

        check_type(self.diag_K_xKK_x_T, self.float_type)
        self.diag_K_xKK_x_T_gpu = allocate_array(self.diag_K_xKK_x_T, self.diag_K_xKK_x_T_gpu)
        #drv.memcpy_htod(self.diag_K_xKK_x_T_gpu, self.diag_K_xKK_x_T)

        check_type(self.variance, self.float_type)
        self.variance_gpu = allocate_array(self.variance, self.variance_gpu)

        check_type(self.feedback, self.float_type)
        self.feedback_gpu = allocate_array(self.feedback, self.feedback_gpu, zeros=False)
        drv.memcpy_htod(self.feedback_gpu, self.feedback)

        check_type(self.mean, self.float_type)
        self.mean_gpu = allocate_array(self.mean, self.mean_gpu)

        check_type(self.ucb, self.float_type)
        self.ucb_gpu = allocate_array(self.ucb, self.ucb_gpu)

    def get_variance(self):
        drv.memcpy_dtoh(self.variance, self.variance_gpu)
        return self.variance

    def get_mean(self):
        drv.memcpy_dtoh(self.mean, self.mean_gpu)
        return self.mean

    def get_ucb(self):
        drv.memcpy_dtoh(self.ucb, self.ucb_gpu)
        return self.ucb

    def increase_matrix_sizes(self, user_feedback, feedback_idx):
        # TODO: Not finished, use set_feedback for now
        self.shown_idx_gpu.free()
        self.feedback_gpu.free()
        self.K_gpu.free()
        self.K_noise_gpu.free()
        self.K_inv_gpu.free()
        self.K_x_gpu.free()
        self.K_xK_gpu.free()

        self.n_shown += 1
        self.n_shown_padded = self.round_up_to_blocksize(self.n_shown)

        self.feedback = pad_vector(self.feedback, self.n_shown - 1, self.n_shown_padded, dtype=self.float_type)
        self.shown_idx = pad_vector(self.shown_idx, self.n_shown - 1, self.n_shown_padded, dtype=self.float_type)
        self.K_noise = pad_vector(self.K_noise, self.n_shown - 1, self.n_shown_padded, dtype=self.float_type)

        self.feedback[self.n_shown - 1] = self.float_type(user_feedback)
        self.shown_idx[self.n_shown - 1] = self.float_type(feedback_idx)
        self.K_noise[self.n_shown - 1] = self.float_type(np.random.random())

        self.shown_idx_gpu = drv.mem_alloc(self.shown_idx.nbytes)
        self.feedback_gpu = drv.mem_alloc(self.feedback.nbytes)
        self.K_noise_gpu = drv.mem_alloc(self.K_noise.nbytes)


    def add_feedback(self, user_feedback, feedback_idx):
        """
        Add a feedback value into the feedback vector and the index of the image for which feedback was given
        to the shown images vector.

        Parameters:
            user_feedback is the feedback value from user (float)
            feedback_idx is the index of the image for which the feedback was given
        """
        # TODO: Add cuda operation success checks?
        drv.memcpy_dtoh(self.feedback, self.feedback_gpu)
        drv.memcpy_dtoh(self.shown_idx, self.shown_idx_gpu)
        if self.n_shown < self.n_shown_padded:  # If padded array has room left, add to it
            self.feedback[self.n_shown] = self.float_type(user_feedback)  # Update feedback vector
            self.shown_idx[self.n_shown] = self.int_type(feedback_idx)  # Update shown idx vector
            self.K_noise[self.n_shown] = self.float_type(np.random.normal(1, 0.1))
            self.n_shown += 1
            drv.memcpy_htod(self.feedback_gpu, self.feedback)
            drv.memcpy_htod(self.shown_idx_gpu, self.shown_idx)
            drv.memcpy_htod(self.K_noise_gpu, self.K_noise)

        else:  # Padded arrays are full, so create new padded arrays one blocksize larger before adding values.
            tmp_feedback = self.feedback.tolist()
            tmp_shown_idx = self.shown_idx.tolist()
            tmp_feedback.append(user_feedback)
            tmp_shown_idx.append(feedback_idx)
            self.set_feedback(tmp_feedback, tmp_shown_idx)

    def round_up_to_blocksize(self, num):
        if num % self.block_size[0] != 0:
            return num + (self.block_size[0] - (num % self.block_size[0]))
        return self.int_type(num)

    def gaussian_process(self, debug=False):
        tmp_K = np.zeros(self.K.shape, dtype=self.float_type)
        drv.memcpy_htod(self.K_gpu, tmp_K)
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
            if len(self.img_features) <= 1000:
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

        tmp_K_xKK_x_T = np.zeros(self.diag_K_xKK_x_T.shape, dtype=self.float_type)
        drv.memcpy_htod(self.diag_K_xKK_x_T_gpu, tmp_K_xKK_x_T)
        self.calc_K_xKK_x_T()
        if debug:
            drv.memcpy_dtoh(self.diag_K_xKK_x_T, self.diag_K_xKK_x_T_gpu)
            K_xKK_x_T_test = np.diag(np.matrix(self.K_xK) * np.matrix(self.K_x).T)
            check_result("K_xKK_x_T", self.diag_K_xKK_x_T, K_xKK_x_T_test)

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

        return self.ucb, self.mean

    def calc_K(self):
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
        grid_size_x = (self.n_predict_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size = (grid_size_x, 1, 1)
        cuda_func = self.cuda_module.get_function("matMulDiag")
        cuda_func(self.K_xK_gpu, self.K_x_gpu, self.diag_K_xKK_x_T_gpu, np.int32(self.n_predict_padded),
                  np.int32(self.n_shown_padded), block=(self.block_size[0], 1, 1), grid=grid_size)

    def calc_variance(self):
        grid_size_x = (self.n_predict_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size = (grid_size_x, 1, 1)
        cuda_func = self.cuda_module.get_function("generate__variance__")
        cuda_func(self.variance_gpu, self.diag_K_xx_gpu, self.diag_K_xKK_x_T_gpu, self.n_predict_padded,
                  block=(self.block_size[0], 1, 1), grid=grid_size)

    def calc_mean(self):
        h = cublas.cublasCreate()
        CUBLAS_OP_N = 0
        alpha = 1.0
        beta = 0.0
        cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, self.n_predict_padded, self.n_shown_padded, alpha,
                           self.feedback_gpu, 1, self.K_xK_gpu, self.n_shown_padded, beta, self.mean_gpu, 1)
        cublas.cublasDestroy(h)

    def calc_UCB(self):
        grid_size_x = (self.n_predict_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size = (grid_size_x, 1, 1)
        cuda_func = self.cuda_module.get_function("generate__UCB__")
        cuda_func(self.ucb_gpu, self.mean_gpu, self.variance_gpu, block=(self.block_size[0], 1, 1), grid=grid_size)


if __name__ == "__main__":
    # Load image features
    feat = np.asfarray(np.load("../../data/cl25000.npy"), dtype="float32")
    feedback = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    feedback_idx = [7025, 14522, 24818, 657, 13478, 4601, 21410, 23775, 10850, 16573, 23139, 17426]
    #feedback = np.array(np.random.random(33))
    #feedback_idx = np.arange(len(feedback))
    gaussianProcess = GaussianProcessGPU(feat)
    gaussianProcess.set_feedback(feedback, feedback_idx)
    gaussianProcess.gaussian_process(debug=True)
    print(np.shape(gaussianProcess.get_mean()))
    print(np.shape(gaussianProcess.get_variance()))
    moo=True
    if moo:
        for i in range(1, 50):
            gaussianProcess.add_feedback(float(i)/10, 1 + i)
            gaussianProcess.gaussian_process(debug=False)
            #print('Mean shape:', np.shape(gaussianProcess.get_mean()))
            #print('Variance shape:', np.shape(gaussianProcess.get_variance()))
