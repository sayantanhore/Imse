__global__ void generate__K__(float *K_gpu, int *shown_gpu, float *feat_gpu, float *K_noise_gpu,
    int shown_size, int feature_size)
{
    // Get co-ordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z >= 1 || y >= shown_size || x >= shown_size) return;

    //atomicAdd(&K_gpu[y * shown_size + x], fabs(feat_gpu[shown_gpu[x] * feature_size + z] -
    //    feat_gpu[shown_gpu[y] * feature_size + z]));

    //__syncthreads();

    //K_gpu[y * shown_size + x] = fdividef(K_gpu[y * shown_size + x], feature_size);

    //__syncthreads();

    //K_gpu[y * shown_size + x] += fabs(feat_gpu[shown_gpu[x] * feature_size + z] - feat_gpu[shown_gpu[y] * feature_size + z]);
    //atomicAdd(&K_gpu[y * shown_size + x], z);
    for (int i = 0; i < feature_size; i++)
    {
	atomicAdd(&K_gpu[y * shown_size + x], fdividef(fabs(feat_gpu[shown_gpu[x] * feature_size + i] - feat_gpu[shown_gpu[y] * feature_size + i]),feature_size));
    }

    if(x == y) {
        K_gpu[y * shown_size +  x] = K_noise_gpu[x];
    }
}

__global__ void generate__K_x__(float *K_x_gpu, int *shown_gpu, int *predict_gpu, float *feat_gpu,
    int shown_size, int prediction_size, int feature_size)
{
    // Get co-ordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z >= 1 || y >= prediction_size || x >= shown_size) return;
    for (int i = 0; i < feature_size; i++)
    {
    atomicAdd(&K_x_gpu[y * shown_size + x], fdividef(fabsf(feat_gpu[shown_gpu[x] * feature_size + i] -
        feat_gpu[predict_gpu[y] * feature_size + i]), feature_size));
    }
    //K_x_gpu[y * shown_size + x] = feat_gpu[predict_gpu[y] * feature_size + 2];
}


__global__ void generate__diag_K_xx__(float *diag_K_xx_gpu, float *K_xx_noise_gpu)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    diag_K_xx_gpu[x] = K_xx_noise_gpu[x];
}

__global__ void matMulDiag(float *A, float *B, float *C, int numRows, int numCols)
// Slow, fix if the speed of this operation becomes an issue
{
    float result = 0.0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > numRows) return;
    for (int i = 0; i < numCols; ++i) {
        C[x] += A[x * numCols + i] * B[x * numCols + i];
    }
}

__global__ void generate__variance__(float *variance_gpu, float *diag_K_xx_gpu, float *diag_K_xKK_x_T_gpu, int length)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > length) return;
    //variance_gpu[x] = sqrtf(fabsf(diag_K_xx_gpu[x] - diag_K_xKK_x_T_gpu[x]));
    variance_gpu[x] = fabsf(diag_K_xx_gpu[x] - diag_K_xKK_x_T_gpu[x]);
}

__global__ void generate__UCB__(float *ucb_gpu, float *mean_gpu, float *variance_gpu)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    ucb_gpu[x] = mean_gpu[x] + variance_gpu[x];
}
