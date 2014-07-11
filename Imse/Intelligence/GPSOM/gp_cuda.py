#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:30:41 2014

@author: sayantan, lasse
"""
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
import scikits.cuda.linalg as linalg
import pycuda.gpuarray as gpuarray
import numpy as np
import scipy.spatial.distance as dist
import sys


def distance(vector1, vector2, metric="manhattan"):
    vdist = 0
    if metric == "manhattan":
        for i in range(len(vector1)):
            vdist += abs(vector1[i] - vector2[i])
    else:
        raise ValueError('Invalid parameter: ' + str(metric))
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


def allocate_gpu(array, type, size_x, size_y, size_z, copy=False):
    """
    Allocate space for array in GPU memory.
    Parameters:
        array is the array to allocate and possibly copy
        type is the expected type of the array for checking
        size_x, size_y and size_z are the unpadded dimensions of the array
        copy determines if the array will be just allocated or also copied to GPU memory (True to copy)
    """
    # TODO: Cuda error checking if operations fail
    check_type(array, type)
    check_dimensions(array, size_x, size_y, size_z)
    array_gpu = drv.mem_alloc(array.nbytes)
    if copy:
        drv.memcpy_htod(array_gpu, array)
    return array_gpu


class GaussianProcessGPU:
    """
    Gaussian process class, which uses GPU for the distance and matrix calculations. This implementation is not
    thoroughly tested, for example large imagesets will probably cause it to fail.

    Constructing GaussianProcessGPU object:
    GaussianProcessGPU(img_features, feedback, img_shown_idx, block_size=(16, 16, 4))

    Parameters:
        img_features is a 2D array, each row containing float features for one image.
        feedback is a vector containing at least one initial observation as a float
        img_shown_idx is a vector containing the indexes of the images for which feedback was given in the feedback
            vector
        block_size is the block size used in the GPU kernel calls. Don't change unless you know what you're doing.
        float_type defines which numpy float type is used (tested with float32, may or may not work with float64)
        int_type defines which numpy int type is used (tested with int32, may or may not work with int64)
        kernel_file is the file from which GPU kernels are read.
    """

    def __init__(self, img_features, block_size=(16, 16, 4), float_type=np.float32,
                 int_type=np.int32, kernel_file='kernels.c', debug=False):
        with open('feat.txt') as infile:
            img_features = np.loadtxt(infile)

        if debug:
            print("Initialized starts")
            img_features = img_features[:500]
        import pycuda.autoinit
        np.set_printoptions(linewidth=500)
        self.float_type = float_type
        self.int_type = int_type
        self.block_size = block_size
        self.n_features = np.size(img_features, 1)  # TODO: Assuming the n_features is divisible by block_size[2], fix
        cuda_source = open(kernel_file, 'r').read()
        if debug:
            print("len cuda_source" + str(len(cuda_source)))
        try:
            self.cuda_module = SourceModule(cuda_source)
        except Exception as e:
            print(e)

        # Inialize variables
        # Pad everything to match block size
        # Add zero row to the beginning of feature matrix for zero padding in cuda operations TODO: is this necessary?
        self.img_features = np.asfarray(np.vstack(([np.zeros(self.n_features)], img_features)), dtype=self.float_type)
        self.n_total = np.size(self.img_features, 0)
        self.n_total_padded = self.round_up_to_blocksize(self.n_total)  # Pad to match block size

        # Allocate GPU memory and copy data, check datatype before each allocation
        # TODO: add dimension checking
        check_type(self.img_features, self.float_type)
        if debug:
            print(self.img_features.shape)
        self.feat_gpu = drv.mem_alloc(self.img_features.nbytes)
        drv.memcpy_htod(self.feat_gpu, self.img_features)

    #def __del__(self):
        #self.context.pop()

    def set_feedback(self, feedback, feedback_indices, debug=False):
        if debug:
            print(feedback)
            print(feedback_indices)

        self.n_shown = np.size(feedback_indices, 0)
        self.n_shown_padded = self.round_up_to_blocksize(self.n_shown)  # Pad to match block size
        self.n_predict = self.int_type(self.n_total - self.n_shown)
        self.n_predict_padded = self.round_up_to_blocksize(self.n_predict)
        self.shown_idx = np.asarray(feedback_indices, dtype=self.int_type)
        self.shown_idx = pad_vector(self.shown_idx, self.n_shown, self.n_shown_padded, dtype=self.int_type)
        self.predict_idx = np.arange(0, self.n_predict, dtype=self.int_type)
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


    def get_variance(self):
        drv.memcpy_dtoh(self.variance, self.variance_gpu)
        return self.variance

    def get_mean(self):
        drv.memcpy_dtoh(self.mean, self.mean_gpu)
        return self.mean

    def get_ucb(self):
        drv.memcpy_dtoh(self.ucb, self.ucb_gpu)
        return self.ucb

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
            self.n_shown += 1
        else:  # Padded arrays are full, so create new padded arrays one blocksize larger before adding values.
            self.feedback_gpu.free()
            self.shown_idx_gpu.free()
            self.n_shown += 1
            self.n_shown_padded = self.round_up_to_blocksize(self.n_shown)
            self.feedback = pad_vector(self.feedback, self.n_shown, self.n_shown_padded, dtype=self.float_type)
            self.shown_idx = pad_vector(self.shown_idx, self.n_shown, self.n_shown_padded, dtype=self.float_type)
            self.feedback[self.n_shown - 1] = user_feedback
            self.shown_idx[self.n_shown - 1] = feedback_idx
        # TODO: Add check for correct values (datatype, dimensions)
        drv.memcpy_htod(self.feedback_gpu, self.feedback)
        drv.memcpy_htod(self.shown_idx_gpu, self.shown_idx)

    def round_up_to_blocksize(self, num):
        if num % self.block_size[0] != 0:
            return num + (self.block_size[0] - (num % self.block_size[0]))
        return self.int_type(num)

    def gaussian_process(self, feedback, img_shown_idx, debug=False):
        if debug:
            with open('feedback.txt') as infile:
                feedback = np.loadtxt(infile)
            with open('feedback_idx.txt') as infile:
                img_shown_idx = np.loadtxt(infile)
                img_shown_idx = [idx % self.n_total for idx in img_shown_idx]
        self.set_feedback(feedback, img_shown_idx, debug=debug)
        self.calc_K()
        if debug:
            K_test_features = np.asfarray([self.img_features[i] for i in self.shown_idx], dtype=self.float_type)
            K_test = dist.cdist(K_test_features, K_test_features, 'cityblock') / self.n_features + np.diag(self.K_noise)
            drv.memcpy_dtoh(self.K, self.K_gpu)
            #check_result('K', self.K[:self.n_shown, :self.n_shown], K_test[:self.n_shown, :self.n_shown])

        self.invert_K()
#        print(self.K)
        self.calc_K_x()
        drv.memcpy_dtoh(self.K_x, self.K_x_gpu)
        if debug:
            K_x_test = np.zeros((self.n_predict_padded, self.n_shown_padded), dtype=self.float_type)
#            for i, idx1 in enumerate(self.predict_idx):
#                for j, idx2 in enumerate(self.shown_idx):
#                    vdist = distance(self.img_features[idx1], self.img_features[idx2]) / len(self.img_features[0])
#                    K_x_test[i][j] = vdist
            #check_result('K_x', self.K_x[:self.n_predict, :self.n_shown], K_x_test[:self.n_predict, :self.n_shown])

        linalg.init()
        K_inv_gpuarr = gpuarray.to_gpu(self.K_inv[:self.n_shown, :self.n_shown].astype(self.float_type))
        K_x_gpuarr = gpuarray.to_gpu(self.K_x[:self.n_predict, :self.n_shown].astype(self.float_type))
        K_xK_gpuarr = linalg.dot(K_x_gpuarr, K_inv_gpuarr)
        self.K_xK = np.zeros_like(self.K_xK)
        self.K_xK[:self.n_predict, :self.n_shown] = K_xK_gpuarr.get()
        drv.memcpy_htod(self.K_xK_gpu, self.K_xK)
        #self.calc_K_xK()
        if debug:
            #drv.memcpy_dtoh(self.K_xK, self.K_xK_gpu)
            K_xK_test = (np.matrix(self.K_x) * np.matrix(self.K_inv))
            check_result('K_xK', self.K_xK[:self.n_predict, :self.n_shown], K_xK_test[:self.n_predict, :self.n_shown])

        self.calc_K_xKK_x_T()
        if debug:
            drv.memcpy_dtoh(self.diag_K_xKK_x_T, self.diag_K_xKK_x_T_gpu)
            K_xKK_x_T_test = np.diag(np.matrix(self.K_xK) * np.matrix(self.K_x).T)
            check_result("K_xKK_x_T", self.diag_K_xKK_x_T, K_xKK_x_T_test)
            print("K_xKK_xT calculated")
#        print(self.diag_K_xKK_x_T)

        self.calc_variance()
        drv.memcpy_dtoh(self.variance, self.variance_gpu)
        if debug:
            variance_test = np.sqrt(
                np.abs(np.subtract(self.diag_K_xx[:self.n_predict], self.diag_K_xKK_x_T[:, :self.n_predict])))
            check_result('Variance', self.variance[:, :self.n_predict], variance_test[:, :self.n_predict])

#        print(self.K_xK.shape)
#        print(self.feedback)
        self.mean = np.dot(self.K_xK, self.feedback)
        self.calc_mean()
        if debug:
            #drv.memcpy_dtoh(self.mean, self.mean_gpu)
            mean_test = np.dot(self.K_xK, self.feedback)  # This is 1D, self.mean is 2D, so slicing for test differs
            check_result('Mean', self.mean[:self.n_predict], mean_test[:self.n_predict])

        if debug:
            # Calculate full result
            feedback = self.feedback[:self.n_shown]
            feedback_indices = self.shown_idx[:self.n_shown]
            predict_indices = self.predict_idx[:self.n_predict]
            data = self.img_features
            test_K_noise = self.K_noise[:self.n_shown]
            test_K_xx = self.diag_K_xx[:self.n_predict]
            test_K_features = np.asfarray([data[i] for i in feedback_indices], dtype=self.float_type)
            test_K = dist.cdist(test_K_features, test_K_features, 'cityblock') / self.n_features + np.diag(test_K_noise)
            test_K_inv = np.linalg.inv(test_K[:self.n_shown, :self.n_shown])
            test_K_x = np.zeros((self.n_predict, self.n_shown), dtype=self.float_type)
            for i, idx1 in enumerate(predict_indices):
                for j, idx2 in enumerate(feedback_indices):
                    vdist = distance(data[idx1], data[idx2]) / len(data[0])
                    test_K_x[i][j] = vdist
            test_K_xK = np.dot(test_K_x, test_K_inv)
            test_K_xKK_x_T = np.diag(np.dot(test_K_xK, test_K_x.T))
            test_variance = np.sqrt(np.abs(np.subtract(test_K_xx, test_K_xKK_x_T)))
            test_mean = np.dot(test_K_xK, feedback)
            print('K_inv allclose:', np.allclose(self.K_inv[:self.n_shown, :self.n_shown], test_K_inv))
            print('K_x allclose:', np.allclose(self.K_x[:self.n_predict, :self.n_shown], test_K_x))
            print('K_xK allclose:', np.allclose(self.K_xK[:self.n_predict, :self.n_shown], test_K_xK))
            print('K_xK true count:', sum(np.isclose(self.K_xK[:self.n_predict, :self.n_shown].flatten(), test_K_xK.flatten())))
            print('variance allclose:', np.allclose(self.variance.flatten()[:self.n_predict], test_variance))
            print('mean allclose:', np.allclose(self.mean[:self.n_predict], test_mean))

            print('Variance isclose True count:', sum(np.isclose(self.variance.flatten()[:self.n_predict], test_variance)))
            print('Mean isclose True count:', sum(np.isclose(self.mean.flatten()[:self.n_predict], test_mean)))
#            print('Mean differences (first 10):', np.subtract(self.mean.flatten()[:10], test_mean[:10]))
#            print(self.mean.flatten()[:10])
#            print(test_mean[:10])

        return self.variance, self.mean

    def calc_K(self):
        """
        """
        grid_size_xy = (self.n_shown_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size_z = (self.n_features + self.block_size[2] - 1) / self.block_size[2]
        grid_size = (grid_size_xy, grid_size_xy, grid_size_z)

        cuda_func = self.cuda_module.get_function("generate__K__")
        cuda_func(self.K_gpu, self.shown_idx_gpu, self.feat_gpu, self.K_noise_gpu, np.int32(self.n_shown_padded),
                  np.int32(self.n_features), block = self.block_size, grid = grid_size)

    def invert_K(self):
        K = np.zeros((self.n_shown_padded, self.n_shown_padded), dtype=self.float_type)
        drv.memcpy_dtoh(K, self.K_gpu)
        tmp = np.linalg.inv(K[:self.n_shown, :self.n_shown])
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
        pass

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
        pass

    def calc_UCB(self):
        grid_size_x = (self.n_predict_padded + self.block_size[0] - 1) / self.block_size[0]
        grid_size = (grid_size_x, 1, 1)
        cuda_func = self.cuda_module.get_function("generate__UCB__")
        cuda_func(self.ucb_gpu, self.mean_gpu, self.variance_gpu, block=(self.block_size[0], 1, 1), grid=grid_size)


if __name__ == "__main__":
    # Load image features
#    feat = np.asfarray(np.load("../../../../data/Data/cl25000.npy"), dtype="float32")
    feedback = np.array(np.random.random(33))

    feedback = np.array(sys.stdin.readline().split('\t'), dtype=np.float32)
    feedback_indices = np.array(sys.stdin.readline().split('\t'), dtype=np.float32)
    debug = False
    gaussianProcess = GaussianProcessGPU(None, debug=debug)

    mean, variance =gaussianProcess.gaussian_process(feedback, feedback_indices, debug=debug)

    for value in mean.flatten():
        sys.stdout.write(str(value) + '\t')
    sys.stdout.write('\n')
    for value in variance.flatten():
        sys.stdout.write(str(value) + '\t')
    sys.stdout.write('\n')
    sys.exit(0)