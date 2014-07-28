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
import numpy as np
import scipy.spatial.distance as dist

import pycuda.gpuarray as gpuarray
import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
import socket


if socket.gethostname() == 'iitti':
    DATA_PATH = '/home/lassetyr/programming/Imse/data/Data/'
    base_path = '/home/lassetyr/programming/Imse/Imse/'
else:
    DATA_PATH = "/ldata/IMSE/data/Data/"
    base_path = '/ldata/IMSE/Imse/Imse/'

def gp_caller(feedback, feedback_indices):
    print "In test....."
    print("Allocation done 28")
    mean, var = gaussian_process(data, feedback, feedback_indices, debug=False)
    print("Allocation done 2999999999")
    #print("Mean : " + str(mean))
    #return "This is test for " + str(feedback)
    print(mean.dtype)
    return mean.tolist(), var.tolist()


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
                     kernel_file= base_path + 'Intelligence/GPSOM/kernels.c', debug=False):
    print("Inside GP")
    if debug:
        print("Initialized starts")
        print("Loading test data")
#        with open('feedback.txt') as infile:
#            feedback = np.loadtxt(infile)
#        with open('feat.txt') as infile:
#            data = np.loadtxt(infile)
#        with open('feedback_idx.txt') as infile:
#            feedback_indices = np.loadtxt(infile)
        np.set_printoptions(linewidth=500)
    import pycuda.autoinit
    print(feedback)
    print(type(feedback_indices))
    #feedback_indices = feedback_indices.tolist()
    print(feedback_indices)
    print(data.shape)
    print(feedback_indices[0])
    print(len(feedback_indices))
    print(type(feedback_indices))
    random_seed = 1
    np.random.seed(random_seed)

    # Write inputs to files
    outfileprefix = 'output/' + str(len(feedback) - 12) + '_'
    outfile_feedback = outfileprefix + 'feedback.npy'
    outfile_feedback_indices = outfileprefix + 'feedback_indices.npy'
    outfile_randomseed = outfileprefix + 'random_seed.npy'
    np.save(outfile_feedback, feedback)
    np.save(outfile_feedback_indices, feedback_indices)
    np.save(outfile_randomseed, random_seed)

    float_type = float_type
    int_type = int_type
    block_size = (16, 16, 4)
    n_features = np.size(data, 1)  # TODO: Assuming the n_features is divisible by block_size[2]
    print("Start from here")
    #print(kernel_file
    cuda_module = open(kernel_file, 'r').read()
    print("len cuda_module" + str(len(cuda_module)))
    try:
        cuda_module = SourceModule(cuda_module)
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
    data_gpu = drv.mem_alloc(data.nbytes)
    drv.memcpy_htod(data_gpu, data)
    print("Allocation done")
    check_type(feedback_indices, int_type)
    print("Allocation done 2")
    feedback_indices_gpu = drv.mem_alloc(feedback_indices.nbytes)
    print("Allocation done 3")
    drv.memcpy_htod(feedback_indices_gpu, feedback_indices)
    print("Allocation done 4")
    check_type(K, float_type)
    print("Allocation done 5")
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
    predict_indices_gpu = drv.mem_alloc(predict_indices.nbytes)
    drv.memcpy_htod(predict_indices_gpu, predict_indices)
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
    print("Allocation done 20")

    calc_K(cuda_module, block_size, n_features, n_feedback_padded, data_gpu, feedback_indices_gpu, K_noise_gpu, K_gpu)
    print("Allocation done 21")
    if debug:
        K_test_features = np.asfarray([data[i] for i in feedback_indices], dtype=float_type)
        K_test = dist.cdist(K_test_features, K_test_features, 'cityblock') / n_features + np.diag(K_noise)
        drv.memcpy_dtoh(K, K_gpu)
        check_result('K', K[:n_feedback, :n_feedback], K_test[:n_feedback, :n_feedback])

    K_inv = invert_K(n_feedback, n_feedback_padded, float_type, K_gpu, K_inv_gpu)
    print("Allocation done 22")

    calc_K_x(cuda_module, block_size, n_feedback_padded, n_predict_padded, n_features, feedback_indices_gpu,
             predict_indices_gpu, data_gpu, K_x_gpu)
    drv.memcpy_dtoh(K_x, K_x_gpu)
    if debug:
        K_x_test = np.zeros((n_predict_padded, n_feedback_padded), dtype=float_type)
#        for i, idx1 in enumerate(predict_indices):
#            for j, idx2 in enumerate(feedback_indices):
#                vdist = distance(data[idx1], data[idx2]) / len(data[0])
#                K_x_test[i][j] = vdist
#        check_result('K_x', K_x[:n_predict, :n_feedback], K_x_test[:n_predict, :n_feedback])

    linalg.init()
    K_inv_gpuarr = gpuarray.to_gpu(K_inv.astype(float_type))
    K_x_gpuarr = gpuarray.to_gpu(K_x.astype(float_type))
    K_xK_gpu = linalg.dot(K_x_gpuarr, K_inv_gpuarr)
    K_xK = K_xK_gpu.get()
    print("Allocation done 23")

    #calc_K_xK()
    if debug:
        #drv.memcpy_dtoh(K_xK, K_xK_gpu)
        K_xK_test = (np.matrix(K_x) * np.matrix(K_inv))
        check_result('K_xK', K_xK[:n_predict, :n_feedback], K_xK_test[:n_predict, :n_feedback])
        print(K_xK.shape)

    calc_K_xKK_x_T(cuda_module, block_size, n_feedback_padded, n_predict_padded, K_xK_gpu, K_x_gpu, diag_K_xKK_x_T_gpu)
    print("Allocation done 24")
    if debug:
        drv.memcpy_dtoh(diag_K_xKK_x_T, diag_K_xKK_x_T_gpu)
        K_xKK_x_T_test = np.diag(np.matrix(K_xK) * np.matrix(K_x).T)
        check_result("K_xKK_x_T", diag_K_xKK_x_T, K_xKK_x_T_test)

    calc_variance(cuda_module, block_size, n_predict_padded, diag_K_xx_gpu, diag_K_xKK_x_T_gpu, variance_gpu)
    print("Allocation done 25")
    if debug:
        drv.memcpy_dtoh(variance, variance_gpu)
        variance_test = np.sqrt(
            np.abs(np.subtract(diag_K_xx[:n_predict], diag_K_xKK_x_T[:, :n_predict])))
        check_result('Variance', variance[:, :n_predict], variance_test[:, :n_predict])

    mean = np.dot(K_xK, feedback)
    print("Allocation done 26")
    if debug:
        #drv.memcpy_dtoh(mean, mean_gpu)
        mean_test = np.dot(K_xK, feedback)
        check_result('Mean', mean[:n_predict], mean_test[:n_predict])

    if debug:
        # Calculate full result
        feedback = feedback[:n_feedback]
        feedback_indices = feedback_indices[:n_feedback]
        predict_indices = predict_indices[:n_predict]
        data = data
        test_K_noise = K_noise[:n_feedback]
        test_K_xx = diag_K_xx[:n_predict]
        test_K_features = np.asfarray([data[i] for i in feedback_indices], dtype=float_type)
        test_K = dist.cdist(test_K_features, test_K_features, 'cityblock') / n_features + np.diag(test_K_noise)
        test_K_inv = np.linalg.inv(test_K[:n_feedback, :n_feedback])
        test_K_x = np.zeros((n_predict, n_feedback), dtype=float_type)
        for i, idx1 in enumerate(predict_indices):
            for j, idx2 in enumerate(feedback_indices):
                vdist = distance(data[idx1], data[idx2]) / len(data[0])
                test_K_x[i][j] = vdist
        test_K_xK = np.dot(test_K_x, test_K_inv)
        test_K_xKK_x_T = np.diag(np.dot(test_K_xK, test_K_x.T))
        test_variance = np.sqrt(np.abs(np.subtract(test_K_xx, test_K_xKK_x_T)))
        test_mean = np.dot(test_K_xK, feedback)

        print(np.allclose(variance.flatten()[:n_predict], test_variance))
        print(np.allclose(mean[:n_predict], test_mean))

        print('Variance isclose True count:', sum(np.isclose(variance.flatten()[:n_predict], test_variance)))
        print('Mean isclose True count:', sum(np.isclose(mean.flatten()[:n_predict], test_mean)))
        print('Mean differences (first 10):', np.subtract(mean.flatten()[:10], test_mean[:10]))
        print(mean.flatten()[:10])
        print(test_mean[:10])
    print("Allocation done 27")
    print(mean)

    # Write results to files for testing

    outfile_mean = outfileprefix + 'mean.npy'
    outfile_variance = outfileprefix + 'variance.npy'
    np.save(outfile_mean, mean)
    np.save(outfile_variance, variance)

    return mean, variance


def calc_K(cuda_module, block_size, n_features, n_feedback_padded, data_gpu, feedback_indices_gpu, K_noise_gpu, K_gpu):
    """
    """
    grid_size_xy = (n_feedback_padded + block_size[0] - 1) / block_size[0]
    grid_size_z = (n_features + block_size[2] - 1) / block_size[2]
    grid_size = (grid_size_xy, grid_size_xy, grid_size_z)

    cuda_func = cuda_module.get_function("generate__K__")
    cuda_func(K_gpu, feedback_indices_gpu, data_gpu, K_noise_gpu, np.int32(n_feedback_padded),
              np.int32(n_features), block=block_size, grid=grid_size)


def invert_K(n_feedback, n_feedback_padded, float_type, K_gpu, K_inv_gpu):
    K = np.zeros((n_feedback_padded, n_feedback_padded), dtype=float_type)
    drv.memcpy_dtoh(K, K_gpu)
    K_inv = np.zeros((n_feedback_padded, n_feedback_padded))
    tmp = np.linalg.inv(K[:n_feedback, :n_feedback])
    K_inv[:tmp.shape[0], :tmp.shape[1]] = tmp
    K_inv = np.array(K_inv, dtype=float_type)
    drv.memcpy_htod(K_inv_gpu, K_inv)
    return K_inv


def calc_K_x(cuda_module, block_size, n_feedback_padded, n_predict_padded, n_features, feedback_indices_gpu,
             predict_indices_gpu, data_gpu, K_x_gpu):
    grid_size_x = (n_feedback_padded + block_size[0] - 1) / block_size[0]
    grid_size_y = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size_z = (n_features + block_size[2] - 1) / block_size[2]
    grid_size = (grid_size_x, grid_size_y, grid_size_z)

    cuda_func = cuda_module.get_function("generate__K_x__")
    cuda_func(K_x_gpu, feedback_indices_gpu, predict_indices_gpu, data_gpu, np.int32(n_feedback_padded),
              np.int32(n_predict_padded), np.int32(n_features), block=block_size, grid=grid_size)


def calc_K_xK():
    pass


def calc_K_xKK_x_T(cuda_module, block_size, n_feedback_padded, n_predict_padded, K_xK_gpu, K_x_gpu, diag_K_xKK_x_T_gpu):
    grid_size_x = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size = (grid_size_x, 1, 1)
    cuda_func = cuda_module.get_function("matMulDiag")
    cuda_func(K_xK_gpu, K_x_gpu, diag_K_xKK_x_T_gpu, np.int32(n_predict_padded),
              np.int32(n_feedback_padded), block=(block_size[0], 1, 1), grid=grid_size)

def calc_variance(cuda_module, block_size, n_predict_padded, diag_K_xx_gpu, diag_K_xKK_x_T_gpu, variance_gpu):
    grid_size_x = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size = (grid_size_x, 1, 1)
    cuda_func = cuda_module.get_function("generate__variance__")
    cuda_func(variance_gpu, diag_K_xx_gpu, diag_K_xKK_x_T_gpu, n_predict_padded,
              block=(block_size[0], 1, 1), grid=grid_size)

def calc_mean():
    pass

def calc_UCB(cuda_module, block_size, n_predict_padded, mean_gpu, variance_gpu, ucb_gpu):
    grid_size_x = (n_predict_padded + block_size[0] - 1) / block_size[0]
    grid_size = (grid_size_x, 1, 1)
    cuda_func = cuda_module.get_function("generate__UCB__")
    cuda_func(ucb_gpu, mean_gpu, variance_gpu, block=(block_size[0], 1, 1), grid=grid_size)


if __name__ == "__main__":
    # Load image features
    #feat = np.asfarray(np.load("../../../../data/Data/cl25000.npy"), dtype="float32")
    #feat = None
    #feedback = np.array(np.random.random(33))

    #print(np.shape(mean))
    #print(np.shape(ucb))
    data = np.asfarray(np.load(DATA_PATH + "cl25000.npy"), dtype="float32")
#    feedback = [0 for i in range(10)]
#    mean, ucb = gaussian_process(feat, feedback, np.arange(len(feedback), dtype="int32"), debug=True)


    server = SimpleXMLRPCServer(("localhost", 8888))
    server.register_function(gp_caller, "gp")
    #server.register_function(gaussian_process, "gp")
    print("Listening at 8888")
    server.serve_forever()


