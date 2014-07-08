#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:30:41 2014

@author: sayantan, lasse
"""
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
import scikits.cuda.cublas as cublas
import numpy as np
import scipy.spatial.distance as dist
from Intelligence.path.Path import FILE_ROOT_PATH


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

def round_up_to_blocksize(num, block_size, int_type):
    if num % block_size[0] != 0:
        return num + (block_size[0] - (num % block_size[0]))
    return int_type(num)

def gaussian_process(data, feedback, feedback_indices, float_type=np.float32, int_type=np.int32,
                     kernel_file=FILE_ROOT_PATH + 'Intelligence/GPSOM/kernels.c', debug=False):
    print("Initialized starts")
    print("Loading test data")
    with open('feedback.txt') as infile:
        feedback = np.loadtxt(infile)
    with open('feat.txt') as infile:
        data = np.loadtxt(infile)
    with open('feedback_idx.txt') as infile:
        feedback_indices = np.loadtxt(infile)
    import pycuda.autoinit
    np.set_printoptions(linewidth=500)
    print(feedback)
    feedback_indices = feedback_indices.tolist()
    print(feedback_indices)
    print(data.shape)
    print(feedback_indices[0])
    print(len(feedback_indices))
    print(type(feedback_indices))
    float_type = float_type
    int_type = int_type
    block_size = (16, 16, 4)
    n_features = np.size(data, 1)  # TODO: Assuming the n_features is divisible by block_size[2]
    print("Start from here")
    #print(kernel_file
    cuda_source = open(kernel_file, 'r').read()
    print("len cuda_source" + str(len(cuda_source)))
    try:
        cuda_module = SourceModule(cuda_source)
    except Exception as e:
        print(e)
    print("Check zero")
    
    # Inialize variables
    # Pad everything to match block size
    # Add zero row to the beginning of feature matrix for zero padding in cuda operations TODO: is this necessary?
    data = np.asfarray(np.vstack(([np.zeros(n_features)], data)), dtype=float_type)
    n_total = np.size(data, 0)
    n_total_padded = round_up_to_blocksize(n_total, block_size, int_type)  # Pad to match block size
    n_feedback = np.size(feedback_indices, 0)
    n_feedback_padded = round_up_to_blocksize(n_feedback, block_size, int_type)  # Pad to match block size
    n_predict = int_type(n_total - n_feedback)
    n_predict_padded = round_up_to_blocksize(n_predict, block_size, int_type)
    feedback_indices = np.asarray(feedback_indices, dtype=int_type)
    feedback_indices = pad_vector(feedback_indices, n_feedback, n_feedback_padded, dtype=int_type)
    predict_indices = np.arange(0, n_predict, dtype=int_type)  # TODO: wut?
    predict_indices = pad_vector(predict_indices, n_predict, n_predict_padded)
    K = np.zeros((n_feedback_padded, n_feedback_padded), dtype=float_type)
    K_x = np.zeros((n_predict_padded, n_feedback_padded), dtype=float_type)
    K_xK = np.zeros((n_predict_padded, n_feedback_padded), dtype=float_type)
    K_noise = cumath.np.random.normal(1, 0.1, n_feedback)  # Generate diagonal noise
    K_noise = pad_vector(K_noise, n_feedback, n_feedback_padded, dtype=float_type)
    K_inv = np.asfarray(K, dtype=float_type)
    diag_K_xx = cumath.np.random.normal(1, 0.1, n_predict)
    diag_K_xx = pad_vector(diag_K_xx, n_predict, n_predict_padded, dtype=float_type)
    diag_K_xKK_x_T = np.zeros((1, n_predict_padded), dtype=float_type)
    variance = np.zeros((1, n_predict_padded), dtype=float_type)
    feedback = np.array(feedback)
    feedback = pad_vector(feedback, n_feedback, n_feedback_padded, dtype=float_type)
    mean = np.zeros((1, n_predict_padded), dtype=float_type)
    ucb = np.zeros((1, n_predict_padded), dtype=float_type)
    print("Check one")
    
    # Allocate GPU memory and copy data, check datatype before each allocation
    # TODO: add dimension checking
    check_type(data, float_type)
    print(data.shape)
    feat_gpu = drv.mem_alloc(data.nbytes)
    drv.memcpy_htod(feat_gpu, data)
    check_type(feedback_indices, int_type)
    feedback_indices_gpu = drv.mem_alloc(feedback_indices.nbytes)
    drv.memcpy_htod(feedback_indices_gpu, feedback_indices)
    check_type(K, float_type)
    K_gpu = drv.mem_alloc(K.nbytes)
    drv.memcpy_htod(K_gpu, K)
    check_type(K_inv, float_type)
    K_inv_gpu = drv.mem_alloc(K_inv.nbytes)
    check_type(K_noise, float_type)
    K_noise_gpu = drv.mem_alloc(K_noise.nbytes)
    drv.memcpy_htod(K_noise_gpu, K_noise)
    check_type(K_x, float_type)
    K_x_gpu = drv.mem_alloc(K_x.nbytes)
    drv.memcpy_htod(K_x_gpu, K_x)
    check_type(predict_indices, int_type)
    predict_idx_gpu = drv.mem_alloc(predict_indices.nbytes)
    drv.memcpy_htod(predict_idx_gpu, predict_indices)
    check_type(K_xK, float_type)
    K_xK_gpu = drv.mem_alloc(K_xK.nbytes)
    check_type(diag_K_xx, float_type)
    diag_K_xx_gpu = drv.mem_alloc(diag_K_xx.nbytes)
    drv.memcpy_htod(diag_K_xx_gpu, diag_K_xx)
    check_type(diag_K_xKK_x_T, float_type)
    diag_K_xKK_x_T_gpu = drv.mem_alloc(diag_K_xKK_x_T.nbytes)
    drv.memcpy_htod(diag_K_xKK_x_T_gpu, diag_K_xKK_x_T)
    check_type(variance, float_type)
    variance_gpu = drv.mem_alloc(variance.nbytes)
    check_type(feedback, float_type)
    feedback_gpu = drv.mem_alloc(feedback.nbytes)
    drv.memcpy_htod(feedback_gpu, feedback)
    check_type(mean, float_type)
    mean_gpu = drv.mem_alloc(mean.nbytes)
    check_type(ucb, float_type)
    ucb_gpu = drv.mem_alloc(ucb.nbytes)

    calc_K()
    if debug:
        K_test_features = np.asfarray([data[i] for i in feedback_indices], dtype=float_type)
        K_test = dist.cdist(K_test_features, K_test_features, 'cityblock') / n_features + np.diag(K_noise)
        drv.memcpy_dtoh(K, K_gpu)
        check_result('K', K[:n_feedback, :n_feedback], K_test[:n_feedback, :n_feedback])

    invert_K()
    calc_K_x()
    if debug:
        drv.memcpy_dtoh(K_x, K_x_gpu)
        K_x_test = np.zeros((n_predict_padded, n_feedback_padded), dtype=float_type)
#            for i, idx1 in enumerate(predict_idx):
#                for j, idx2 in enumerate(shown_idx):
#                    vdist = distance(img_features[idx1], img_features[idx2]) / len(img_features[0])
#                    K_x_test[i][j] = vdist
        #check_result('K_x', K_x[:n_predict, :n_feedback], K_x_test[:n_predict, :n_feedback])

    calc_K_xK()
    if debug:
        drv.memcpy_dtoh(K_xK, K_xK_gpu)
        K_xK_test = (np.matrix(K_x) * np.matrix(K_inv))
        check_result('K_xK', K_xK[:n_predict, :n_feedback], K_xK_test[:n_predict, :n_feedback])

    calc_K_xKK_x_T()
    if debug:
        drv.memcpy_dtoh(diag_K_xKK_x_T, diag_K_xKK_x_T_gpu)
        K_xKK_x_T_test = np.diag(np.matrix(K_xK) * np.matrix(K_x).T)
        check_result("K_xKK_x_T", diag_K_xKK_x_T, K_xKK_x_T_test)
    print("K_xKK_xT")

    calc_variance()
    if debug:
        drv.memcpy_dtoh(variance, variance_gpu)
        variance_test = np.sqrt(
            np.abs(np.subtract(diag_K_xx[:n_predict], diag_K_xKK_x_T[:, :n_predict])))
        check_result('Variance', variance[:, :n_predict], variance_test[:, :n_predict])

    calc_mean()
    if debug:
        drv.memcpy_dtoh(mean, mean_gpu)
        mean_test = np.dot(K_xK, feedback)  # This is 1D, mean is 2D, so slicing for test differs
        check_result('Mean', mean[:, :n_predict], mean_test[:n_predict])

    calc_UCB()
    if debug:
        drv.memcpy_dtoh(ucb, ucb_gpu)
        ucb_test = np.add(mean, variance)  # The array shapes differ a bit, so slicing is different
        check_result('UCB', ucb[:n_predict, :], ucb_test[:n_predict])
    print("Returning from GP-CUDA")
    return ucb, mean


def calc_K(cuda_module, block_size, n_features, n_feedback_padded, data_gpu, feedback_indices_gpu, K_noise_gpu, K_gpu):
    """
    """
    grid_size_xy = (n_feedback_padded + block_size[0] - 1) / block_size[0]
    grid_size_z = (n_features + block_size[2] - 1) / block_size[2]
    grid_size = (grid_size_xy, grid_size_xy, grid_size_z)

    cuda_func = cuda_module.get_function("generate__K__")
    cuda_func(K_gpu, feedback_indices_gpu, data_gpu, K_noise_gpu, np.int32(n_feedback_padded),
              np.int32(n_features), block=block_size, grid=grid_size)

def invert_K():
    K = np.zeros((n_feedback_padded, n_feedback_padded), dtype=float_type)
    drv.memcpy_dtoh(K, K_gpu)
    tmp = cumath.np.linalg.inv(K[:n_feedback, :n_feedback])
    K_inv[:tmp.shape[0], :tmp.shape[1]] = tmp
    drv.memcpy_htod(K_inv_gpu, K_inv)

def calc_K_x():
    grid_size_x = (n_feedback_padded + block_size[0] - 1) / block_size[0]
    grid_size_y = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size_z = (n_features + block_size[2] - 1) / block_size[2]
    grid_size = (grid_size_x, grid_size_y, grid_size_z)

    cuda_func = cuda_module.get_function("generate__K_x__")
    cuda_func(K_x_gpu, shown_idx_gpu, predict_idx_gpu, feat_gpu, np.int32(n_feedback_padded),
              np.int32(n_predict_padded), np.int32(n_features), block=block_size, grid=grid_size)

def calc_K_xK():
    h = cublas.cublasCreate()
    #cublas.cublasCheckStatus(h)
    CUBLAS_OP_N = 0
    alpha = 1.0
    beta = 0.0
    drv.memcpy_dtoh(K_inv, K_inv_gpu)
    drv.memcpy_dtoh(K_x, K_x_gpu)
    print('K_inv.shape' + str(K_inv.shape))
    print('K_x.shape' + str(K_x.shape))
    print('K_xK.shape' + str(K_xK.shape))
    print(n_feedback_padded)
    print(n_predict_padded)

    cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n_feedback_padded, n_predict_padded, n_feedback_padded,
                       alpha, K_inv_gpu, n_feedback_padded, K_x_gpu, n_feedback_padded, beta,
                       K_xK_gpu, n_feedback_padded)
    print('moo')
    cublas.cublasDestroy(h)

def calc_K_xKK_x_T():
    grid_size_x = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size = (grid_size_x, 1, 1)
    cuda_func = cuda_module.get_function("matMulDiag")
    cuda_func(K_xK_gpu, K_x_gpu, diag_K_xKK_x_T_gpu, np.int32(n_predict_padded),
              np.int32(n_feedback_padded), block=(block_size[0], 1, 1), grid=grid_size)

def calc_variance():
    grid_size_x = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size = (grid_size_x, 1, 1)
    cuda_func = cuda_module.get_function("generate__variance__")
    cuda_func(variance_gpu, diag_K_xx_gpu, diag_K_xKK_x_T_gpu, n_predict_padded,
              block=(block_size[0], 1, 1), grid=grid_size)

def calc_mean():
    h = cublas.cublasCreate()
    CUBLAS_OP_N = 0
    alpha = 1.0
    beta = 0.0
    cublas.cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, n_predict_padded, n_feedback_padded, alpha,
                       feedback_gpu, 1, K_xK_gpu, n_feedback_padded, beta, mean_gpu, 1)
    cublas.cublasDestroy(h)

def calc_UCB():
    grid_size_x = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size = (grid_size_x, 1, 1)
    cuda_func = cuda_module.get_function("generate__UCB__")
    cuda_func(ucb_gpu, mean_gpu, variance_gpu, block=(block_size[0], 1, 1), grid=grid_size)


if __name__ == "__main__":
    # Load image features
    feat = np.asfarray(np.load("../../../../data/Data/cl25000.npy"), dtype="float32")
    feedback = np.array(np.random.random(33))
    mean, ucb = gaussian_process(feat, feedback, np.arange(len(feedback), dtype="int32"))
    print(np.shape(mean))
    print(np.shape(ucb))

