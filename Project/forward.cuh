
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define _K 7
#define _C 12
#define _W 72
#define _H 72
#define _M 24
#define _B 10000
#define BLOCK_SIZE 16
#define BLOCK_SIZE_C1 16 //16
#define BLOCK_SIZE_C12 32 // 32
#define NUM_STREAM_C1 200
#define NUM_STREAM_C12 200
#define CPU_BATCH 100
__constant__ float filter[_M*_C*_K*_K];

__global__ void forward_kernel0(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working
   //printf("B: %d M: %d C: %d H: %d W: %d K: %d\n", B, M, C, H, W, K);
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockDim.x * blockIdx.x + threadIdx.x;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++)
                {
                    y4d(b, m, h, w) = 0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q); 
                }
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel1(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockDim.x * blockIdx.x + threadIdx.x;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++)
                {
                    y4d(b, m, h, w) = 0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q); 
                }
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel2(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockDim.x * blockIdx.x + threadIdx.x;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++)
                {
                    float res = 0.0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                res += x4d(b, c, h + p, w + q) * k4d(m, c, p, q); 
                    y4d(b, m, h, w) = res;
                }
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel3(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockIdx.x;
    int m = blockIdx.y;

    int tmp = (W_out + BLOCK_SIZE-1) / BLOCK_SIZE;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE + threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE + threadIdx.x;

    if (h < H_out && w < W_out)
    {
        float res = 0.0;
        for (int c = 0; c < C; c++) 
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    res += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
        y4d(b, m, h, w) = res;
    }

#undef y4d
#undef x4d
#undef k4d
}

/* __global__ void forward_kernel4(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockIdx.x;
    int m = blockIdx.y;

    int tmp = (W_out + BLOCK_SIZE-1) / BLOCK_SIZE;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE + threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE + threadIdx.x;
    
    //__shared__ float x_shared[_K + BLOCK_SIZE-1][_K + BLOCK_SIZE-1];
    __shared__ float x_shared[_C][_K + BLOCK_SIZE-1][_K + BLOCK_SIZE-1];

    for (int c = 0; c < C; c++)
    {
        //if (h < H && w < W)
            x_shared[c][threadIdx.y][threadIdx.x] = x4d(b, c, h, w);
        //else
        //    x_shared[c][threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    if (threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE && h < H_out && w < W_out)
    {
        float res = 0.0;

        for (int c = 0; c < C; c++)
        {       
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    res += x_shared[c][threadIdx.y+p][threadIdx.x+q] * k4d(m, c, p, q);
        }
        y4d(b, m, h, w) = res;
    }

#undef y4d
#undef x4d
#undef k4d
} */

/* __global__ void forward_kernel5(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockIdx.x;
    int m = blockIdx.y;

    int tmp = (W_out + BLOCK_SIZE-1) / BLOCK_SIZE;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE+ threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE + threadIdx.x;

    __shared__ float x_shared[BLOCK_SIZE+_K-1][BLOCK_SIZE+_K-1];
    //__shared__ float k_shared[_K][_K];
    
    float res = 0.0;
    for (int c = 0; c < C; c++)
    {
        x_shared[threadIdx.y][threadIdx.x] = x4d(b, c, h, w);
        //if (threadIdx.y < K && threadIdx.x < K)
        //    k_shared[threadIdx.y][threadIdx.x] = k4d(m, c, threadIdx.y, threadIdx.x);
        __syncthreads();

        if (threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE && h < H_out && w < W_out)
        {
            for (int p = 0; p < K; p++)
            {
                res += x_shared[threadIdx.y+p][threadIdx.x] * k4d(m, c, p, 0);
                res += x_shared[threadIdx.y+p][threadIdx.x+1] * k4d(m, c, p, 1);
                res += x_shared[threadIdx.y+p][threadIdx.x+2] * k4d(m, c, p, 2);
                res += x_shared[threadIdx.y+p][threadIdx.x+3] * k4d(m, c, p, 3);
                res += x_shared[threadIdx.y+p][threadIdx.x+4] * k4d(m, c, p, 4);
                res += x_shared[threadIdx.y+p][threadIdx.x+5] * k4d(m, c, p, 5);
                res += x_shared[threadIdx.y+p][threadIdx.x+6] * k4d(m, c, p, 6);
            }
        }
        __syncthreads();
    }

    if (threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE && h < H_out && w < W_out)
        y4d(b, m, h, w) = res;

#undef y4d
#undef x4d
#undef k4d
} */

/* __global__ void forward_kernel5_C1(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int y4d_ref1 = M * H_out * W_out;
    const int y4d_ref2 = H_out * W_out;
    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define y4d(i3, i2, i1, i0) y[(i3) * y4d_ref1 + (i2) * y4d_ref2 + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * x4d_ref1 + (i2) * x4d_ref2 + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * k4d_ref1 + (i2) * k4d_ref2 + (i1) * (K) + i0]

    int b = blockIdx.x;
    int m = blockIdx.y;

    int tmp = (W_out + BLOCK_SIZE-1) / BLOCK_SIZE;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE+ threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE + threadIdx.x;

    __shared__ float x_shared[_C][BLOCK_SIZE+_K-1][BLOCK_SIZE+_K-1];
    //__shared__ float k_shared[_K][_K];

    for (int c = 0; c < C; c++)
        x_shared[c][threadIdx.y][threadIdx.x] = x4d(b, c, h, w);
    __syncthreads();
    
    float res = 0.0;
    if ( threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE && h < H_out && w < W_out)
    {
        int c = 0;
            res += x_shared[c][threadIdx.y][threadIdx.x] * k4d(m, c, 0, 0);
            res += x_shared[c][threadIdx.y][threadIdx.x+1] * k4d(m, c, 0, 1);
            res += x_shared[c][threadIdx.y][threadIdx.x+2] * k4d(m, c, 0, 2);
            res += x_shared[c][threadIdx.y][threadIdx.x+3] * k4d(m, c, 0, 3);
            res += x_shared[c][threadIdx.y][threadIdx.x+4] * k4d(m, c, 0, 4);
            res += x_shared[c][threadIdx.y][threadIdx.x+5] * k4d(m, c, 0, 5);
            res += x_shared[c][threadIdx.y][threadIdx.x+6] * k4d(m, c, 0, 6);

            res += x_shared[c][threadIdx.y+1][threadIdx.x] * k4d(m, c, 1, 0);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+1] * k4d(m, c, 1, 1);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+2] * k4d(m, c, 1, 2);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+3] * k4d(m, c, 1, 3);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+4] * k4d(m, c, 1, 4);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+5] * k4d(m, c, 1, 5);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+6] * k4d(m, c, 1, 6);

            res += x_shared[c][threadIdx.y+2][threadIdx.x] * k4d(m, c, 2, 0);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+1] * k4d(m, c, 2, 1);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+2] * k4d(m, c, 2, 2);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+3] * k4d(m, c, 2, 3);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+4] * k4d(m, c, 2, 4);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+5] * k4d(m, c, 2, 5);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+6] * k4d(m, c, 2, 6);

            res += x_shared[c][threadIdx.y+3][threadIdx.x] * k4d(m, c, 3, 0);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+1] * k4d(m, c, 3, 1);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+2] * k4d(m, c, 3, 2);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+3] * k4d(m, c, 3, 3);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+4] * k4d(m, c, 3, 4);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+5] * k4d(m, c, 3, 5);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+6] * k4d(m, c, 3, 6);

            res += x_shared[c][threadIdx.y+4][threadIdx.x] * k4d(m, c, 4, 0);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+1] * k4d(m, c, 4, 1);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+2] * k4d(m, c, 4, 2);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+3] * k4d(m, c, 4, 3);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+4] * k4d(m, c, 4, 4);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+5] * k4d(m, c, 4, 5);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+6] * k4d(m, c, 4, 6);

            res += x_shared[c][threadIdx.y+5][threadIdx.x] * k4d(m, c, 5, 0);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+1] * k4d(m, c, 5, 1);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+2] * k4d(m, c, 5, 2);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+3] * k4d(m, c, 5, 3);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+4] * k4d(m, c, 5, 4);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+5] * k4d(m, c, 5, 5);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+6] * k4d(m, c, 5, 6);

            res += x_shared[c][threadIdx.y+6][threadIdx.x] * k4d(m, c, 6, 0);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+1] * k4d(m, c, 6, 1);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+2] * k4d(m, c, 6, 2);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+3] * k4d(m, c, 6, 3);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+4] * k4d(m, c, 6, 4);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+5] * k4d(m, c, 6, 5);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+6] * k4d(m, c, 6, 6);
            
            y4d(b, m, h, w) = res;
        }
        
#undef y4d
#undef x4d
#undef k4d
} */

/* __global__ void forward_kernel5_C12(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int y4d_ref1 = M * H_out * W_out;
    const int y4d_ref2 = H_out * W_out;
    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define y4d(i3, i2, i1, i0) y[(i3) * y4d_ref1 + (i2) * y4d_ref2 + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * x4d_ref1 + (i2) * x4d_ref2 + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * k4d_ref1 + (i2) * k4d_ref2 + (i1) * (K) + i0]

    int b = blockIdx.x;
    int m = blockIdx.y;

    int tmp = (W_out + BLOCK_SIZE-1) / BLOCK_SIZE;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE+ threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE + threadIdx.x;

    __shared__ float x_shared[_C][BLOCK_SIZE+_K-1][BLOCK_SIZE+_K-1];

    for (int c = 0; c < C; c++)
        x_shared[c][threadIdx.y][threadIdx.x] = x4d(b, c, h, w);
    __syncthreads();
    
    float res = 0.0;
    if ( threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE && h < H_out && w < W_out)
    {
        for (int c = 0; c < C; c++)
        {
            res += x_shared[c][threadIdx.y][threadIdx.x] * k4d(m, c, 0, 0);
            res += x_shared[c][threadIdx.y][threadIdx.x+1] * k4d(m, c, 0, 1);
            res += x_shared[c][threadIdx.y][threadIdx.x+2] * k4d(m, c, 0, 2);
            res += x_shared[c][threadIdx.y][threadIdx.x+3] * k4d(m, c, 0, 3);
            res += x_shared[c][threadIdx.y][threadIdx.x+4] * k4d(m, c, 0, 4);
            res += x_shared[c][threadIdx.y][threadIdx.x+5] * k4d(m, c, 0, 5);
            res += x_shared[c][threadIdx.y][threadIdx.x+6] * k4d(m, c, 0, 6);

            res += x_shared[c][threadIdx.y+1][threadIdx.x] * k4d(m, c, 1, 0);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+1] * k4d(m, c, 1, 1);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+2] * k4d(m, c, 1, 2);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+3] * k4d(m, c, 1, 3);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+4] * k4d(m, c, 1, 4);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+5] * k4d(m, c, 1, 5);
            res += x_shared[c][threadIdx.y+1][threadIdx.x+6] * k4d(m, c, 1, 6);

            res += x_shared[c][threadIdx.y+2][threadIdx.x] * k4d(m, c, 2, 0);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+1] * k4d(m, c, 2, 1);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+2] * k4d(m, c, 2, 2);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+3] * k4d(m, c, 2, 3);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+4] * k4d(m, c, 2, 4);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+5] * k4d(m, c, 2, 5);
            res += x_shared[c][threadIdx.y+2][threadIdx.x+6] * k4d(m, c, 2, 6);

            res += x_shared[c][threadIdx.y+3][threadIdx.x] * k4d(m, c, 3, 0);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+1] * k4d(m, c, 3, 1);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+2] * k4d(m, c, 3, 2);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+3] * k4d(m, c, 3, 3);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+4] * k4d(m, c, 3, 4);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+5] * k4d(m, c, 3, 5);
            res += x_shared[c][threadIdx.y+3][threadIdx.x+6] * k4d(m, c, 3, 6);

            res += x_shared[c][threadIdx.y+4][threadIdx.x] * k4d(m, c, 4, 0);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+1] * k4d(m, c, 4, 1);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+2] * k4d(m, c, 4, 2);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+3] * k4d(m, c, 4, 3);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+4] * k4d(m, c, 4, 4);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+5] * k4d(m, c, 4, 5);
            res += x_shared[c][threadIdx.y+4][threadIdx.x+6] * k4d(m, c, 4, 6);

            res += x_shared[c][threadIdx.y+5][threadIdx.x] * k4d(m, c, 5, 0);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+1] * k4d(m, c, 5, 1);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+2] * k4d(m, c, 5, 2);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+3] * k4d(m, c, 5, 3);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+4] * k4d(m, c, 5, 4);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+5] * k4d(m, c, 5, 5);
            res += x_shared[c][threadIdx.y+5][threadIdx.x+6] * k4d(m, c, 5, 6);

            res += x_shared[c][threadIdx.y+6][threadIdx.x] * k4d(m, c, 6, 0);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+1] * k4d(m, c, 6, 1);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+2] * k4d(m, c, 6, 2);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+3] * k4d(m, c, 6, 3);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+4] * k4d(m, c, 6, 4);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+5] * k4d(m, c, 6, 5);
            res += x_shared[c][threadIdx.y+6][threadIdx.x+6] * k4d(m, c, 6, 6);
        }
        y4d(b, m, h, w) = res;
    }
        
#undef y4d
#undef x4d
#undef k4d
} */

/* __device__ inline void innerLoopUnroll(const float *x, float & res, const int C, const int H, 
                    const int W, const int K, const int & h, const int & w, const int & c)
{
    int b = blockIdx.x;
    int m = blockIdx.y;

    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define x4d(i3, i2, i1, i0) x[(i3) * (x4d_ref1) + (i2) * (x4d_ref2) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (k4d_ref1) + (i2) * (k4d_ref2) + (i1) * (K) + i0]
    
    //float res = 0.0f;
    res += x4d(b, c, h, w) * k4d(m, c, 0, 0);
    res += x4d(b, c, h, w+1) * k4d(m, c, 0, 1);
    res += x4d(b, c, h, w+2) * k4d(m, c, 0, 2);
    res += x4d(b, c, h, w+3) * k4d(m, c, 0, 3);
    res += x4d(b, c, h, w+4) * k4d(m, c, 0, 4);
    res += x4d(b, c, h, w+5) * k4d(m, c, 0, 5);
    res += x4d(b, c, h, w+6) * k4d(m, c, 0, 6);

    res += x4d(b, c, h+1, w) * k4d(m, c, 1, 0);
    res += x4d(b, c, h+1, w+1) * k4d(m, c, 1, 1);
    res += x4d(b, c, h+1, w+2) * k4d(m, c, 1, 2);
    res += x4d(b, c, h+1, w+3) * k4d(m, c, 1, 3);
    res += x4d(b, c, h+1, w+4) * k4d(m, c, 1, 4);
    res += x4d(b, c, h+1, w+5) * k4d(m, c, 1, 5);
    res += x4d(b, c, h+1, w+6) * k4d(m, c, 1, 6);

    res += x4d(b, c, h+2, w) * k4d(m, c, 2, 0);
    res += x4d(b, c, h+2, w+1) * k4d(m, c, 2, 1);
    res += x4d(b, c, h+2, w+2) * k4d(m, c, 2, 2);
    res += x4d(b, c, h+2, w+3) * k4d(m, c, 2, 3);
    res += x4d(b, c, h+2, w+4) * k4d(m, c, 2, 4);
    res += x4d(b, c, h+2, w+5) * k4d(m, c, 2, 5);
    res += x4d(b, c, h+2, w+6) * k4d(m, c, 2, 6);

    res += x4d(b, c, h+3, w) * k4d(m, c, 3, 0);
    res += x4d(b, c, h+3, w+1) * k4d(m, c, 3, 1);
    res += x4d(b, c, h+3, w+2) * k4d(m, c, 3, 2);
    res += x4d(b, c, h+3, w+3) * k4d(m, c, 3, 3);
    res += x4d(b, c, h+3, w+4) * k4d(m, c, 3, 4);
    res += x4d(b, c, h+3, w+5) * k4d(m, c, 3, 5);
    res += x4d(b, c, h+3, w+6) * k4d(m, c, 3, 6);

    res += x4d(b, c, h+4, w) * k4d(m, c, 4, 0);
    res += x4d(b, c, h+4, w+1) * k4d(m, c, 4, 1);
    res += x4d(b, c, h+4, w+2) * k4d(m, c, 4, 2);
    res += x4d(b, c, h+4, w+3) * k4d(m, c, 4, 3);
    res += x4d(b, c, h+4, w+4) * k4d(m, c, 4, 4);
    res += x4d(b, c, h+4, w+5) * k4d(m, c, 4, 5);
    res += x4d(b, c, h+4, w+6) * k4d(m, c, 4, 6);

    res += x4d(b, c, h+5, w) * k4d(m, c, 5, 0);
    res += x4d(b, c, h+5, w+1) * k4d(m, c, 5, 1);
    res += x4d(b, c, h+5, w+2) * k4d(m, c, 5, 2);
    res += x4d(b, c, h+5, w+3) * k4d(m, c, 5, 3);
    res += x4d(b, c, h+5, w+4) * k4d(m, c, 5, 4);
    res += x4d(b, c, h+5, w+5) * k4d(m, c, 5, 5);
    res += x4d(b, c, h+5, w+6) * k4d(m, c, 5, 6);

    res += x4d(b, c, h+6, w) * k4d(m, c, 6, 0);
    res += x4d(b, c, h+6, w+1) * k4d(m, c, 6, 1);
    res += x4d(b, c, h+6, w+2) * k4d(m, c, 6, 2);
    res += x4d(b, c, h+6, w+3) * k4d(m, c, 6, 3);
    res += x4d(b, c, h+6, w+4) * k4d(m, c, 6, 4);
    res += x4d(b, c, h+6, w+5) * k4d(m, c, 6, 5);
    res += x4d(b, c, h+6, w+6) * k4d(m, c, 6, 6);

#undef x4d
#undef k4d
    //return res;
} */

__device__ inline void loopUnroll(const float *x, float & res, const int C, const int H, 
                    const int W, const int K, const int & h, const int & w)
{
    int b = blockIdx.x;
    int m = blockIdx.y;

    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define x4d(i3, i2, i1, i0) x[(i3) * (x4d_ref1) + (i2) * (x4d_ref2) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (k4d_ref1) + (i2) * (k4d_ref2) + (i1) * (K) + i0]
    
    //float res = 0.0f;
    for (int c = 0; c < C; c+=4) 
    {
        res += x4d(b, c, h, w) * k4d(m, c, 0, 0);
        res += x4d(b, c, h, w+1) * k4d(m, c, 0, 1);
        res += x4d(b, c, h, w+2) * k4d(m, c, 0, 2);
        res += x4d(b, c, h, w+3) * k4d(m, c, 0, 3);
        res += x4d(b, c, h, w+4) * k4d(m, c, 0, 4);
        res += x4d(b, c, h, w+5) * k4d(m, c, 0, 5);
        res += x4d(b, c, h, w+6) * k4d(m, c, 0, 6);

        res += x4d(b, c, h+1, w) * k4d(m, c, 1, 0);
        res += x4d(b, c, h+1, w+1) * k4d(m, c, 1, 1);
        res += x4d(b, c, h+1, w+2) * k4d(m, c, 1, 2);
        res += x4d(b, c, h+1, w+3) * k4d(m, c, 1, 3);
        res += x4d(b, c, h+1, w+4) * k4d(m, c, 1, 4);
        res += x4d(b, c, h+1, w+5) * k4d(m, c, 1, 5);
        res += x4d(b, c, h+1, w+6) * k4d(m, c, 1, 6);

        res += x4d(b, c, h+2, w) * k4d(m, c, 2, 0);
        res += x4d(b, c, h+2, w+1) * k4d(m, c, 2, 1);
        res += x4d(b, c, h+2, w+2) * k4d(m, c, 2, 2);
        res += x4d(b, c, h+2, w+3) * k4d(m, c, 2, 3);
        res += x4d(b, c, h+2, w+4) * k4d(m, c, 2, 4);
        res += x4d(b, c, h+2, w+5) * k4d(m, c, 2, 5);
        res += x4d(b, c, h+2, w+6) * k4d(m, c, 2, 6);

        res += x4d(b, c, h+3, w) * k4d(m, c, 3, 0);
        res += x4d(b, c, h+3, w+1) * k4d(m, c, 3, 1);
        res += x4d(b, c, h+3, w+2) * k4d(m, c, 3, 2);
        res += x4d(b, c, h+3, w+3) * k4d(m, c, 3, 3);
        res += x4d(b, c, h+3, w+4) * k4d(m, c, 3, 4);
        res += x4d(b, c, h+3, w+5) * k4d(m, c, 3, 5);
        res += x4d(b, c, h+3, w+6) * k4d(m, c, 3, 6);

        res += x4d(b, c, h+4, w) * k4d(m, c, 4, 0);
        res += x4d(b, c, h+4, w+1) * k4d(m, c, 4, 1);
        res += x4d(b, c, h+4, w+2) * k4d(m, c, 4, 2);
        res += x4d(b, c, h+4, w+3) * k4d(m, c, 4, 3);
        res += x4d(b, c, h+4, w+4) * k4d(m, c, 4, 4);
        res += x4d(b, c, h+4, w+5) * k4d(m, c, 4, 5);
        res += x4d(b, c, h+4, w+6) * k4d(m, c, 4, 6);

        res += x4d(b, c, h+5, w) * k4d(m, c, 5, 0);
        res += x4d(b, c, h+5, w+1) * k4d(m, c, 5, 1);
        res += x4d(b, c, h+5, w+2) * k4d(m, c, 5, 2);
        res += x4d(b, c, h+5, w+3) * k4d(m, c, 5, 3);
        res += x4d(b, c, h+5, w+4) * k4d(m, c, 5, 4);
        res += x4d(b, c, h+5, w+5) * k4d(m, c, 5, 5);
        res += x4d(b, c, h+5, w+6) * k4d(m, c, 5, 6);

        res += x4d(b, c, h+6, w) * k4d(m, c, 6, 0);
        res += x4d(b, c, h+6, w+1) * k4d(m, c, 6, 1);
        res += x4d(b, c, h+6, w+2) * k4d(m, c, 6, 2);
        res += x4d(b, c, h+6, w+3) * k4d(m, c, 6, 3);
        res += x4d(b, c, h+6, w+4) * k4d(m, c, 6, 4);
        res += x4d(b, c, h+6, w+5) * k4d(m, c, 6, 5);
        res += x4d(b, c, h+6, w+6) * k4d(m, c, 6, 6);

        res += x4d(b, c+1, h, w) * k4d(m, c+1, 0, 0);
        res += x4d(b, c+1, h, w+1) * k4d(m, c+1, 0, 1);
        res += x4d(b, c+1, h, w+2) * k4d(m, c+1, 0, 2);
        res += x4d(b, c+1, h, w+3) * k4d(m, c+1, 0, 3);
        res += x4d(b, c+1, h, w+4) * k4d(m, c+1, 0, 4);
        res += x4d(b, c+1, h, w+5) * k4d(m, c+1, 0, 5);
        res += x4d(b, c+1, h, w+6) * k4d(m, c+1, 0, 6);

        res += x4d(b, c+1, h+1, w) * k4d(m, c+1, 1, 0);
        res += x4d(b, c+1, h+1, w+1) * k4d(m, c+1, 1, 1);
        res += x4d(b, c+1, h+1, w+2) * k4d(m, c+1, 1, 2);
        res += x4d(b, c+1, h+1, w+3) * k4d(m, c+1, 1, 3);
        res += x4d(b, c+1, h+1, w+4) * k4d(m, c+1, 1, 4);
        res += x4d(b, c+1, h+1, w+5) * k4d(m, c+1, 1, 5);
        res += x4d(b, c+1, h+1, w+6) * k4d(m, c+1, 1, 6);

        res += x4d(b, c+1, h+2, w) * k4d(m, c+1, 2, 0);
        res += x4d(b, c+1, h+2, w+1) * k4d(m, c+1, 2, 1);
        res += x4d(b, c+1, h+2, w+2) * k4d(m, c+1, 2, 2);
        res += x4d(b, c+1, h+2, w+3) * k4d(m, c+1, 2, 3);
        res += x4d(b, c+1, h+2, w+4) * k4d(m, c+1, 2, 4);
        res += x4d(b, c+1, h+2, w+5) * k4d(m, c+1, 2, 5);
        res += x4d(b, c+1, h+2, w+6) * k4d(m, c+1, 2, 6);

        res += x4d(b, c+1, h+3, w) * k4d(m, c+1, 3, 0);
        res += x4d(b, c+1, h+3, w+1) * k4d(m, c+1, 3, 1);
        res += x4d(b, c+1, h+3, w+2) * k4d(m, c+1, 3, 2);
        res += x4d(b, c+1, h+3, w+3) * k4d(m, c+1, 3, 3);
        res += x4d(b, c+1, h+3, w+4) * k4d(m, c+1, 3, 4);
        res += x4d(b, c+1, h+3, w+5) * k4d(m, c+1, 3, 5);
        res += x4d(b, c+1, h+3, w+6) * k4d(m, c+1, 3, 6);

        res += x4d(b, c+1, h+4, w) * k4d(m, c+1, 4, 0);
        res += x4d(b, c+1, h+4, w+1) * k4d(m, c+1, 4, 1);
        res += x4d(b, c+1, h+4, w+2) * k4d(m, c+1, 4, 2);
        res += x4d(b, c+1, h+4, w+3) * k4d(m, c+1, 4, 3);
        res += x4d(b, c+1, h+4, w+4) * k4d(m, c+1, 4, 4);
        res += x4d(b, c+1, h+4, w+5) * k4d(m, c+1, 4, 5);
        res += x4d(b, c+1, h+4, w+6) * k4d(m, c+1, 4, 6);

        res += x4d(b, c+1, h+5, w) * k4d(m, c+1, 5, 0);
        res += x4d(b, c+1, h+5, w+1) * k4d(m, c+1, 5, 1);
        res += x4d(b, c+1, h+5, w+2) * k4d(m, c+1, 5, 2);
        res += x4d(b, c+1, h+5, w+3) * k4d(m, c+1, 5, 3);
        res += x4d(b, c+1, h+5, w+4) * k4d(m, c+1, 5, 4);
        res += x4d(b, c+1, h+5, w+5) * k4d(m, c+1, 5, 5);
        res += x4d(b, c+1, h+5, w+6) * k4d(m, c+1, 5, 6);

        res += x4d(b, c+1, h+6, w) * k4d(m, c+1, 6, 0);
        res += x4d(b, c+1, h+6, w+1) * k4d(m, c+1, 6, 1);
        res += x4d(b, c+1, h+6, w+2) * k4d(m, c+1, 6, 2);
        res += x4d(b, c+1, h+6, w+3) * k4d(m, c+1, 6, 3);
        res += x4d(b, c+1, h+6, w+4) * k4d(m, c+1, 6, 4);
        res += x4d(b, c+1, h+6, w+5) * k4d(m, c+1, 6, 5);
        res += x4d(b, c+1, h+6, w+6) * k4d(m, c+1, 6, 6);


        res += x4d(b, c+2, h, w) * k4d(m, c+2, 0, 0);
        res += x4d(b, c+2, h, w+1) * k4d(m, c+2, 0, 1);
        res += x4d(b, c+2, h, w+2) * k4d(m, c+2, 0, 2);
        res += x4d(b, c+2, h, w+3) * k4d(m, c+2, 0, 3);
        res += x4d(b, c+2, h, w+4) * k4d(m, c+2, 0, 4);
        res += x4d(b, c+2, h, w+5) * k4d(m, c+2, 0, 5);
        res += x4d(b, c+2, h, w+6) * k4d(m, c+2, 0, 6);

        res += x4d(b, c+2, h+1, w) * k4d(m, c+2, 1, 0);
        res += x4d(b, c+2, h+1, w+1) * k4d(m, c+2, 1, 1);
        res += x4d(b, c+2, h+1, w+2) * k4d(m, c+2, 1, 2);
        res += x4d(b, c+2, h+1, w+3) * k4d(m, c+2, 1, 3);
        res += x4d(b, c+2, h+1, w+4) * k4d(m, c+2, 1, 4);
        res += x4d(b, c+2, h+1, w+5) * k4d(m, c+2, 1, 5);
        res += x4d(b, c+2, h+1, w+6) * k4d(m, c+2, 1, 6);

        res += x4d(b, c+2, h+2, w) * k4d(m, c+2, 2, 0);
        res += x4d(b, c+2, h+2, w+1) * k4d(m, c+2, 2, 1);
        res += x4d(b, c+2, h+2, w+2) * k4d(m, c+2, 2, 2);
        res += x4d(b, c+2, h+2, w+3) * k4d(m, c+2, 2, 3);
        res += x4d(b, c+2, h+2, w+4) * k4d(m, c+2, 2, 4);
        res += x4d(b, c+2, h+2, w+5) * k4d(m, c+2, 2, 5);
        res += x4d(b, c+2, h+2, w+6) * k4d(m, c+2, 2, 6);

        res += x4d(b, c+2, h+3, w) * k4d(m, c+2, 3, 0);
        res += x4d(b, c+2, h+3, w+1) * k4d(m, c+2, 3, 1);
        res += x4d(b, c+2, h+3, w+2) * k4d(m, c+2, 3, 2);
        res += x4d(b, c+2, h+3, w+3) * k4d(m, c+2, 3, 3);
        res += x4d(b, c+2, h+3, w+4) * k4d(m, c+2, 3, 4);
        res += x4d(b, c+2, h+3, w+5) * k4d(m, c+2, 3, 5);
        res += x4d(b, c+2, h+3, w+6) * k4d(m, c+2, 3, 6);

        res += x4d(b, c+2, h+4, w) * k4d(m, c+2, 4, 0);
        res += x4d(b, c+2, h+4, w+1) * k4d(m, c+2, 4, 1);
        res += x4d(b, c+2, h+4, w+2) * k4d(m, c+2, 4, 2);
        res += x4d(b, c+2, h+4, w+3) * k4d(m, c+2, 4, 3);
        res += x4d(b, c+2, h+4, w+4) * k4d(m, c+2, 4, 4);
        res += x4d(b, c+2, h+4, w+5) * k4d(m, c+2, 4, 5);
        res += x4d(b, c+2, h+4, w+6) * k4d(m, c+2, 4, 6);

        res += x4d(b, c+2, h+5, w) * k4d(m, c+2, 5, 0);
        res += x4d(b, c+2, h+5, w+1) * k4d(m, c+2, 5, 1);
        res += x4d(b, c+2, h+5, w+2) * k4d(m, c+2, 5, 2);
        res += x4d(b, c+2, h+5, w+3) * k4d(m, c+2, 5, 3);
        res += x4d(b, c+2, h+5, w+4) * k4d(m, c+2, 5, 4);
        res += x4d(b, c+2, h+5, w+5) * k4d(m, c+2, 5, 5);
        res += x4d(b, c+2, h+5, w+6) * k4d(m, c+2, 5, 6);
        
        res += x4d(b, c+2, h+6, w) * k4d(m, c+2, 6, 0);
        res += x4d(b, c+2, h+6, w+1) * k4d(m, c+2, 6, 1);
        res += x4d(b, c+2, h+6, w+2) * k4d(m, c+2, 6, 2);
        res += x4d(b, c+2, h+6, w+3) * k4d(m, c+2, 6, 3);
        res += x4d(b, c+2, h+6, w+4) * k4d(m, c+2, 6, 4);
        res += x4d(b, c+2, h+6, w+5) * k4d(m, c+2, 6, 5);
        res += x4d(b, c+2, h+6, w+6) * k4d(m, c+2, 6, 6);

        
        res += x4d(b, c+3, h, w) * k4d(m, c+3, 0, 0);
        res += x4d(b, c+3, h, w+1) * k4d(m, c+3, 0, 1);
        res += x4d(b, c+3, h, w+2) * k4d(m, c+3, 0, 2);
        res += x4d(b, c+3, h, w+3) * k4d(m, c+3, 0, 3);
        res += x4d(b, c+3, h, w+4) * k4d(m, c+3, 0, 4);
        res += x4d(b, c+3, h, w+5) * k4d(m, c+3, 0, 5);
        res += x4d(b, c+3, h, w+6) * k4d(m, c+3, 0, 6);

        res += x4d(b, c+3, h+1, w) * k4d(m, c+3, 1, 0);
        res += x4d(b, c+3, h+1, w+1) * k4d(m, c+3, 1, 1);
        res += x4d(b, c+3, h+1, w+2) * k4d(m, c+3, 1, 2);
        res += x4d(b, c+3, h+1, w+3) * k4d(m, c+3, 1, 3);
        res += x4d(b, c+3, h+1, w+4) * k4d(m, c+3, 1, 4);
        res += x4d(b, c+3, h+1, w+5) * k4d(m, c+3, 1, 5);
        res += x4d(b, c+3, h+1, w+6) * k4d(m, c+3, 1, 6);

        res += x4d(b, c+3, h+2, w) * k4d(m, c+3, 2, 0);
        res += x4d(b, c+3, h+2, w+1) * k4d(m, c+3, 2, 1);
        res += x4d(b, c+3, h+2, w+2) * k4d(m, c+3, 2, 2);
        res += x4d(b, c+3, h+2, w+3) * k4d(m, c+3, 2, 3);
        res += x4d(b, c+3, h+2, w+4) * k4d(m, c+3, 2, 4);
        res += x4d(b, c+3, h+2, w+5) * k4d(m, c+3, 2, 5);
        res += x4d(b, c+3, h+2, w+6) * k4d(m, c+3, 2, 6);

        res += x4d(b, c+3, h+3, w) * k4d(m, c+3, 3, 0);
        res += x4d(b, c+3, h+3, w+1) * k4d(m, c+3, 3, 1);
        res += x4d(b, c+3, h+3, w+2) * k4d(m, c+3, 3, 2);
        res += x4d(b, c+3, h+3, w+3) * k4d(m, c+3, 3, 3);
        res += x4d(b, c+3, h+3, w+4) * k4d(m, c+3, 3, 4);
        res += x4d(b, c+3, h+3, w+5) * k4d(m, c+3, 3, 5);
        res += x4d(b, c+3, h+3, w+6) * k4d(m, c+3, 3, 6);

        res += x4d(b, c+3, h+4, w) * k4d(m, c+3, 4, 0);
        res += x4d(b, c+3, h+4, w+1) * k4d(m, c+3, 4, 1);
        res += x4d(b, c+3, h+4, w+2) * k4d(m, c+3, 4, 2);
        res += x4d(b, c+3, h+4, w+3) * k4d(m, c+3, 4, 3);
        res += x4d(b, c+3, h+4, w+4) * k4d(m, c+3, 4, 4);
        res += x4d(b, c+3, h+4, w+5) * k4d(m, c+3, 4, 5);
        res += x4d(b, c+3, h+4, w+6) * k4d(m, c+3, 4, 6);

        res += x4d(b, c+3, h+5, w) * k4d(m, c+3, 5, 0);
        res += x4d(b, c+3, h+5, w+1) * k4d(m, c+3, 5, 1);
        res += x4d(b, c+3, h+5, w+2) * k4d(m, c+3, 5, 2);
        res += x4d(b, c+3, h+5, w+3) * k4d(m, c+3, 5, 3);
        res += x4d(b, c+3, h+5, w+4) * k4d(m, c+3, 5, 4);
        res += x4d(b, c+3, h+5, w+5) * k4d(m, c+3, 5, 5);
        res += x4d(b, c+3, h+5, w+6) * k4d(m, c+3, 5, 6);
        
        res += x4d(b, c+3, h+6, w) * k4d(m, c+3, 6, 0);
        res += x4d(b, c+3, h+6, w+1) * k4d(m, c+3, 6, 1);
        res += x4d(b, c+3, h+6, w+2) * k4d(m, c+3, 6, 2);
        res += x4d(b, c+3, h+6, w+3) * k4d(m, c+3, 6, 3);
        res += x4d(b, c+3, h+6, w+4) * k4d(m, c+3, 6, 4);
        res += x4d(b, c+3, h+6, w+5) * k4d(m, c+3, 6, 5);
        res += x4d(b, c+3, h+6, w+6) * k4d(m, c+3, 6, 6);
    }
#undef x4d
#undef k4d
    //return res;
}

__global__ void forward_kernel6_C1(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int y4d_ref1 = M * H_out * W_out;
    const int y4d_ref2 = H_out * W_out;
    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define y4d(i3, i2, i1, i0) y[(i3) * (y4d_ref1) + (i2) * (y4d_ref2) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (x4d_ref1) + (i2) * (x4d_ref2) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (k4d_ref1) + (i2) * (k4d_ref2) + (i1) * (K) + i0]
/* #define y4d(i3, i2, i1, i0) y[__mul24(i3, __mul24(M, __mul24(H_out, W_out))) + __mul24(i2, __mul24(H_out, W_out)) + __mul24(i1, W_out) + i0]
#define x4d(i3, i2, i1, i0) x[__mul24(i3, __mul24(C, __mul24(H, W))) + __mul24(i2, __mul24(H, W)) + __mul24(i1, W) + i0]
#define k4d(i3, i2, i1, i0) filter[__mul24(i3, __mul24(C, __mul24(K, K))) + __mul24(i2, __mul24(K, K)) + __mul24(i1, K) + i0] */

    int b = blockIdx.x;
    int m = blockIdx.y;

    int tmp = (W_out + BLOCK_SIZE_C1-1) / BLOCK_SIZE_C1;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE_C1 + threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE_C1 + threadIdx.x;
    /* int tmp = (W_out + BLOCK_SIZE-1) >> 4;
    int h = (( blockIdx.z / tmp) << 4) + threadIdx.y;
    int w = (( blockIdx.z % tmp) << 4) + threadIdx.x; */

    if (h < H_out && w < W_out)
    {
        float res = 0.0f;
        int c = 0;
        res += x4d(b, c, h, w) * k4d(m, c, 0, 0);
        res += x4d(b, c, h, w+1) * k4d(m, c, 0, 1);
        res += x4d(b, c, h, w+2) * k4d(m, c, 0, 2);
        res += x4d(b, c, h, w+3) * k4d(m, c, 0, 3);
        res += x4d(b, c, h, w+4) * k4d(m, c, 0, 4);
        res += x4d(b, c, h, w+5) * k4d(m, c, 0, 5);
        res += x4d(b, c, h, w+6) * k4d(m, c, 0, 6);

        res += x4d(b, c, h+1, w) * k4d(m, c, 1, 0);
        res += x4d(b, c, h+1, w+1) * k4d(m, c, 1, 1);
        res += x4d(b, c, h+1, w+2) * k4d(m, c, 1, 2);
        res += x4d(b, c, h+1, w+3) * k4d(m, c, 1, 3);
        res += x4d(b, c, h+1, w+4) * k4d(m, c, 1, 4);
        res += x4d(b, c, h+1, w+5) * k4d(m, c, 1, 5);
        res += x4d(b, c, h+1, w+6) * k4d(m, c, 1, 6);

        res += x4d(b, c, h+2, w) * k4d(m, c, 2, 0);
        res += x4d(b, c, h+2, w+1) * k4d(m, c, 2, 1);
        res += x4d(b, c, h+2, w+2) * k4d(m, c, 2, 2);
        res += x4d(b, c, h+2, w+3) * k4d(m, c, 2, 3);
        res += x4d(b, c, h+2, w+4) * k4d(m, c, 2, 4);
        res += x4d(b, c, h+2, w+5) * k4d(m, c, 2, 5);
        res += x4d(b, c, h+2, w+6) * k4d(m, c, 2, 6);

        res += x4d(b, c, h+3, w) * k4d(m, c, 3, 0);
        res += x4d(b, c, h+3, w+1) * k4d(m, c, 3, 1);
        res += x4d(b, c, h+3, w+2) * k4d(m, c, 3, 2);
        res += x4d(b, c, h+3, w+3) * k4d(m, c, 3, 3);
        res += x4d(b, c, h+3, w+4) * k4d(m, c, 3, 4);
        res += x4d(b, c, h+3, w+5) * k4d(m, c, 3, 5);
        res += x4d(b, c, h+3, w+6) * k4d(m, c, 3, 6);

        res += x4d(b, c, h+4, w) * k4d(m, c, 4, 0);
        res += x4d(b, c, h+4, w+1) * k4d(m, c, 4, 1);
        res += x4d(b, c, h+4, w+2) * k4d(m, c, 4, 2);
        res += x4d(b, c, h+4, w+3) * k4d(m, c, 4, 3);
        res += x4d(b, c, h+4, w+4) * k4d(m, c, 4, 4);
        res += x4d(b, c, h+4, w+5) * k4d(m, c, 4, 5);
        res += x4d(b, c, h+4, w+6) * k4d(m, c, 4, 6);

        res += x4d(b, c, h+5, w) * k4d(m, c, 5, 0);
        res += x4d(b, c, h+5, w+1) * k4d(m, c, 5, 1);
        res += x4d(b, c, h+5, w+2) * k4d(m, c, 5, 2);
        res += x4d(b, c, h+5, w+3) * k4d(m, c, 5, 3);
        res += x4d(b, c, h+5, w+4) * k4d(m, c, 5, 4);
        res += x4d(b, c, h+5, w+5) * k4d(m, c, 5, 5);
        res += x4d(b, c, h+5, w+6) * k4d(m, c, 5, 6);

        res += x4d(b, c, h+6, w) * k4d(m, c, 6, 0);
        res += x4d(b, c, h+6, w+1) * k4d(m, c, 6, 1);
        res += x4d(b, c, h+6, w+2) * k4d(m, c, 6, 2);
        res += x4d(b, c, h+6, w+3) * k4d(m, c, 6, 3);
        res += x4d(b, c, h+6, w+4) * k4d(m, c, 6, 4);
        res += x4d(b, c, h+6, w+5) * k4d(m, c, 6, 5);
        res += x4d(b, c, h+6, w+6) * k4d(m, c, 6, 6);
        
        y4d(b, m, h, w) = res;
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel6_C12(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int y4d_ref1 = M * H_out * W_out;
    const int y4d_ref2 = H_out * W_out;
    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define y4d(i3, i2, i1, i0) y[(i3) * (y4d_ref1) + (i2) * (y4d_ref2) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (x4d_ref1) + (i2) * (x4d_ref2) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (k4d_ref1) + (i2) * (k4d_ref2) + (i1) * (K) + i0]
/* #define y4d(i3, i2, i1, i0) y[__mul24(i3, __mul24(M, __mul24(H_out, W_out))) + __mul24(i2, __mul24(H_out, W_out)) + __mul24(i1, W_out) + i0]
#define x4d(i3, i2, i1, i0) x[__mul24(i3, __mul24(C, __mul24(H, W))) + __mul24(i2, __mul24(H, W)) + __mul24(i1, W) + i0]
#define k4d(i3, i2, i1, i0) filter[__mul24(i3, __mul24(C, __mul24(K, K))) + __mul24(i2, __mul24(K, K)) + __mul24(i1, K) + i0] */

    int b = blockIdx.x;
    int m = blockIdx.y;

    int tmp = (W_out + BLOCK_SIZE_C12-1) / BLOCK_SIZE_C12;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE_C12 + threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE_C12 + threadIdx.x;
    /* int tmp = (W_out + BLOCK_SIZE-1) >> 5;
    int h = (( blockIdx.z / tmp) << 5) + threadIdx.y;
    int w = (( blockIdx.z % tmp) << 5) + threadIdx.x; */

    if (h < H_out && w < W_out)
    {
        float res = 0.0f;
        for (int c = 0; c < C; c+=4) 
        {
            res += x4d(b, c, h, w) * k4d(m, c, 0, 0);
            res += x4d(b, c, h, w+1) * k4d(m, c, 0, 1);
            res += x4d(b, c, h, w+2) * k4d(m, c, 0, 2);
            res += x4d(b, c, h, w+3) * k4d(m, c, 0, 3);
            res += x4d(b, c, h, w+4) * k4d(m, c, 0, 4);
            res += x4d(b, c, h, w+5) * k4d(m, c, 0, 5);
            res += x4d(b, c, h, w+6) * k4d(m, c, 0, 6);

            res += x4d(b, c, h+1, w) * k4d(m, c, 1, 0);
            res += x4d(b, c, h+1, w+1) * k4d(m, c, 1, 1);
            res += x4d(b, c, h+1, w+2) * k4d(m, c, 1, 2);
            res += x4d(b, c, h+1, w+3) * k4d(m, c, 1, 3);
            res += x4d(b, c, h+1, w+4) * k4d(m, c, 1, 4);
            res += x4d(b, c, h+1, w+5) * k4d(m, c, 1, 5);
            res += x4d(b, c, h+1, w+6) * k4d(m, c, 1, 6);

            res += x4d(b, c, h+2, w) * k4d(m, c, 2, 0);
            res += x4d(b, c, h+2, w+1) * k4d(m, c, 2, 1);
            res += x4d(b, c, h+2, w+2) * k4d(m, c, 2, 2);
            res += x4d(b, c, h+2, w+3) * k4d(m, c, 2, 3);
            res += x4d(b, c, h+2, w+4) * k4d(m, c, 2, 4);
            res += x4d(b, c, h+2, w+5) * k4d(m, c, 2, 5);
            res += x4d(b, c, h+2, w+6) * k4d(m, c, 2, 6);

            res += x4d(b, c, h+3, w) * k4d(m, c, 3, 0);
            res += x4d(b, c, h+3, w+1) * k4d(m, c, 3, 1);
            res += x4d(b, c, h+3, w+2) * k4d(m, c, 3, 2);
            res += x4d(b, c, h+3, w+3) * k4d(m, c, 3, 3);
            res += x4d(b, c, h+3, w+4) * k4d(m, c, 3, 4);
            res += x4d(b, c, h+3, w+5) * k4d(m, c, 3, 5);
            res += x4d(b, c, h+3, w+6) * k4d(m, c, 3, 6);

            res += x4d(b, c, h+4, w) * k4d(m, c, 4, 0);
            res += x4d(b, c, h+4, w+1) * k4d(m, c, 4, 1);
            res += x4d(b, c, h+4, w+2) * k4d(m, c, 4, 2);
            res += x4d(b, c, h+4, w+3) * k4d(m, c, 4, 3);
            res += x4d(b, c, h+4, w+4) * k4d(m, c, 4, 4);
            res += x4d(b, c, h+4, w+5) * k4d(m, c, 4, 5);
            res += x4d(b, c, h+4, w+6) * k4d(m, c, 4, 6);

            res += x4d(b, c, h+5, w) * k4d(m, c, 5, 0);
            res += x4d(b, c, h+5, w+1) * k4d(m, c, 5, 1);
            res += x4d(b, c, h+5, w+2) * k4d(m, c, 5, 2);
            res += x4d(b, c, h+5, w+3) * k4d(m, c, 5, 3);
            res += x4d(b, c, h+5, w+4) * k4d(m, c, 5, 4);
            res += x4d(b, c, h+5, w+5) * k4d(m, c, 5, 5);
            res += x4d(b, c, h+5, w+6) * k4d(m, c, 5, 6);

            res += x4d(b, c, h+6, w) * k4d(m, c, 6, 0);
            res += x4d(b, c, h+6, w+1) * k4d(m, c, 6, 1);
            res += x4d(b, c, h+6, w+2) * k4d(m, c, 6, 2);
            res += x4d(b, c, h+6, w+3) * k4d(m, c, 6, 3);
            res += x4d(b, c, h+6, w+4) * k4d(m, c, 6, 4);
            res += x4d(b, c, h+6, w+5) * k4d(m, c, 6, 5);
            res += x4d(b, c, h+6, w+6) * k4d(m, c, 6, 6);

            res += x4d(b, c+1, h, w) * k4d(m, c+1, 0, 0);
            res += x4d(b, c+1, h, w+1) * k4d(m, c+1, 0, 1);
            res += x4d(b, c+1, h, w+2) * k4d(m, c+1, 0, 2);
            res += x4d(b, c+1, h, w+3) * k4d(m, c+1, 0, 3);
            res += x4d(b, c+1, h, w+4) * k4d(m, c+1, 0, 4);
            res += x4d(b, c+1, h, w+5) * k4d(m, c+1, 0, 5);
            res += x4d(b, c+1, h, w+6) * k4d(m, c+1, 0, 6);

            res += x4d(b, c+1, h+1, w) * k4d(m, c+1, 1, 0);
            res += x4d(b, c+1, h+1, w+1) * k4d(m, c+1, 1, 1);
            res += x4d(b, c+1, h+1, w+2) * k4d(m, c+1, 1, 2);
            res += x4d(b, c+1, h+1, w+3) * k4d(m, c+1, 1, 3);
            res += x4d(b, c+1, h+1, w+4) * k4d(m, c+1, 1, 4);
            res += x4d(b, c+1, h+1, w+5) * k4d(m, c+1, 1, 5);
            res += x4d(b, c+1, h+1, w+6) * k4d(m, c+1, 1, 6);

            res += x4d(b, c+1, h+2, w) * k4d(m, c+1, 2, 0);
            res += x4d(b, c+1, h+2, w+1) * k4d(m, c+1, 2, 1);
            res += x4d(b, c+1, h+2, w+2) * k4d(m, c+1, 2, 2);
            res += x4d(b, c+1, h+2, w+3) * k4d(m, c+1, 2, 3);
            res += x4d(b, c+1, h+2, w+4) * k4d(m, c+1, 2, 4);
            res += x4d(b, c+1, h+2, w+5) * k4d(m, c+1, 2, 5);
            res += x4d(b, c+1, h+2, w+6) * k4d(m, c+1, 2, 6);

            res += x4d(b, c+1, h+3, w) * k4d(m, c+1, 3, 0);
            res += x4d(b, c+1, h+3, w+1) * k4d(m, c+1, 3, 1);
            res += x4d(b, c+1, h+3, w+2) * k4d(m, c+1, 3, 2);
            res += x4d(b, c+1, h+3, w+3) * k4d(m, c+1, 3, 3);
            res += x4d(b, c+1, h+3, w+4) * k4d(m, c+1, 3, 4);
            res += x4d(b, c+1, h+3, w+5) * k4d(m, c+1, 3, 5);
            res += x4d(b, c+1, h+3, w+6) * k4d(m, c+1, 3, 6);

            res += x4d(b, c+1, h+4, w) * k4d(m, c+1, 4, 0);
            res += x4d(b, c+1, h+4, w+1) * k4d(m, c+1, 4, 1);
            res += x4d(b, c+1, h+4, w+2) * k4d(m, c+1, 4, 2);
            res += x4d(b, c+1, h+4, w+3) * k4d(m, c+1, 4, 3);
            res += x4d(b, c+1, h+4, w+4) * k4d(m, c+1, 4, 4);
            res += x4d(b, c+1, h+4, w+5) * k4d(m, c+1, 4, 5);
            res += x4d(b, c+1, h+4, w+6) * k4d(m, c+1, 4, 6);

            res += x4d(b, c+1, h+5, w) * k4d(m, c+1, 5, 0);
            res += x4d(b, c+1, h+5, w+1) * k4d(m, c+1, 5, 1);
            res += x4d(b, c+1, h+5, w+2) * k4d(m, c+1, 5, 2);
            res += x4d(b, c+1, h+5, w+3) * k4d(m, c+1, 5, 3);
            res += x4d(b, c+1, h+5, w+4) * k4d(m, c+1, 5, 4);
            res += x4d(b, c+1, h+5, w+5) * k4d(m, c+1, 5, 5);
            res += x4d(b, c+1, h+5, w+6) * k4d(m, c+1, 5, 6);

            res += x4d(b, c+1, h+6, w) * k4d(m, c+1, 6, 0);
            res += x4d(b, c+1, h+6, w+1) * k4d(m, c+1, 6, 1);
            res += x4d(b, c+1, h+6, w+2) * k4d(m, c+1, 6, 2);
            res += x4d(b, c+1, h+6, w+3) * k4d(m, c+1, 6, 3);
            res += x4d(b, c+1, h+6, w+4) * k4d(m, c+1, 6, 4);
            res += x4d(b, c+1, h+6, w+5) * k4d(m, c+1, 6, 5);
            res += x4d(b, c+1, h+6, w+6) * k4d(m, c+1, 6, 6);


            res += x4d(b, c+2, h, w) * k4d(m, c+2, 0, 0);
            res += x4d(b, c+2, h, w+1) * k4d(m, c+2, 0, 1);
            res += x4d(b, c+2, h, w+2) * k4d(m, c+2, 0, 2);
            res += x4d(b, c+2, h, w+3) * k4d(m, c+2, 0, 3);
            res += x4d(b, c+2, h, w+4) * k4d(m, c+2, 0, 4);
            res += x4d(b, c+2, h, w+5) * k4d(m, c+2, 0, 5);
            res += x4d(b, c+2, h, w+6) * k4d(m, c+2, 0, 6);

            res += x4d(b, c+2, h+1, w) * k4d(m, c+2, 1, 0);
            res += x4d(b, c+2, h+1, w+1) * k4d(m, c+2, 1, 1);
            res += x4d(b, c+2, h+1, w+2) * k4d(m, c+2, 1, 2);
            res += x4d(b, c+2, h+1, w+3) * k4d(m, c+2, 1, 3);
            res += x4d(b, c+2, h+1, w+4) * k4d(m, c+2, 1, 4);
            res += x4d(b, c+2, h+1, w+5) * k4d(m, c+2, 1, 5);
            res += x4d(b, c+2, h+1, w+6) * k4d(m, c+2, 1, 6);

            res += x4d(b, c+2, h+2, w) * k4d(m, c+2, 2, 0);
            res += x4d(b, c+2, h+2, w+1) * k4d(m, c+2, 2, 1);
            res += x4d(b, c+2, h+2, w+2) * k4d(m, c+2, 2, 2);
            res += x4d(b, c+2, h+2, w+3) * k4d(m, c+2, 2, 3);
            res += x4d(b, c+2, h+2, w+4) * k4d(m, c+2, 2, 4);
            res += x4d(b, c+2, h+2, w+5) * k4d(m, c+2, 2, 5);
            res += x4d(b, c+2, h+2, w+6) * k4d(m, c+2, 2, 6);

            res += x4d(b, c+2, h+3, w) * k4d(m, c+2, 3, 0);
            res += x4d(b, c+2, h+3, w+1) * k4d(m, c+2, 3, 1);
            res += x4d(b, c+2, h+3, w+2) * k4d(m, c+2, 3, 2);
            res += x4d(b, c+2, h+3, w+3) * k4d(m, c+2, 3, 3);
            res += x4d(b, c+2, h+3, w+4) * k4d(m, c+2, 3, 4);
            res += x4d(b, c+2, h+3, w+5) * k4d(m, c+2, 3, 5);
            res += x4d(b, c+2, h+3, w+6) * k4d(m, c+2, 3, 6);

            res += x4d(b, c+2, h+4, w) * k4d(m, c+2, 4, 0);
            res += x4d(b, c+2, h+4, w+1) * k4d(m, c+2, 4, 1);
            res += x4d(b, c+2, h+4, w+2) * k4d(m, c+2, 4, 2);
            res += x4d(b, c+2, h+4, w+3) * k4d(m, c+2, 4, 3);
            res += x4d(b, c+2, h+4, w+4) * k4d(m, c+2, 4, 4);
            res += x4d(b, c+2, h+4, w+5) * k4d(m, c+2, 4, 5);
            res += x4d(b, c+2, h+4, w+6) * k4d(m, c+2, 4, 6);

            res += x4d(b, c+2, h+5, w) * k4d(m, c+2, 5, 0);
            res += x4d(b, c+2, h+5, w+1) * k4d(m, c+2, 5, 1);
            res += x4d(b, c+2, h+5, w+2) * k4d(m, c+2, 5, 2);
            res += x4d(b, c+2, h+5, w+3) * k4d(m, c+2, 5, 3);
            res += x4d(b, c+2, h+5, w+4) * k4d(m, c+2, 5, 4);
            res += x4d(b, c+2, h+5, w+5) * k4d(m, c+2, 5, 5);
            res += x4d(b, c+2, h+5, w+6) * k4d(m, c+2, 5, 6);
            
            res += x4d(b, c+2, h+6, w) * k4d(m, c+2, 6, 0);
            res += x4d(b, c+2, h+6, w+1) * k4d(m, c+2, 6, 1);
            res += x4d(b, c+2, h+6, w+2) * k4d(m, c+2, 6, 2);
            res += x4d(b, c+2, h+6, w+3) * k4d(m, c+2, 6, 3);
            res += x4d(b, c+2, h+6, w+4) * k4d(m, c+2, 6, 4);
            res += x4d(b, c+2, h+6, w+5) * k4d(m, c+2, 6, 5);
            res += x4d(b, c+2, h+6, w+6) * k4d(m, c+2, 6, 6);

            
            res += x4d(b, c+3, h, w) * k4d(m, c+3, 0, 0);
            res += x4d(b, c+3, h, w+1) * k4d(m, c+3, 0, 1);
            res += x4d(b, c+3, h, w+2) * k4d(m, c+3, 0, 2);
            res += x4d(b, c+3, h, w+3) * k4d(m, c+3, 0, 3);
            res += x4d(b, c+3, h, w+4) * k4d(m, c+3, 0, 4);
            res += x4d(b, c+3, h, w+5) * k4d(m, c+3, 0, 5);
            res += x4d(b, c+3, h, w+6) * k4d(m, c+3, 0, 6);

            res += x4d(b, c+3, h+1, w) * k4d(m, c+3, 1, 0);
            res += x4d(b, c+3, h+1, w+1) * k4d(m, c+3, 1, 1);
            res += x4d(b, c+3, h+1, w+2) * k4d(m, c+3, 1, 2);
            res += x4d(b, c+3, h+1, w+3) * k4d(m, c+3, 1, 3);
            res += x4d(b, c+3, h+1, w+4) * k4d(m, c+3, 1, 4);
            res += x4d(b, c+3, h+1, w+5) * k4d(m, c+3, 1, 5);
            res += x4d(b, c+3, h+1, w+6) * k4d(m, c+3, 1, 6);

            res += x4d(b, c+3, h+2, w) * k4d(m, c+3, 2, 0);
            res += x4d(b, c+3, h+2, w+1) * k4d(m, c+3, 2, 1);
            res += x4d(b, c+3, h+2, w+2) * k4d(m, c+3, 2, 2);
            res += x4d(b, c+3, h+2, w+3) * k4d(m, c+3, 2, 3);
            res += x4d(b, c+3, h+2, w+4) * k4d(m, c+3, 2, 4);
            res += x4d(b, c+3, h+2, w+5) * k4d(m, c+3, 2, 5);
            res += x4d(b, c+3, h+2, w+6) * k4d(m, c+3, 2, 6);

            res += x4d(b, c+3, h+3, w) * k4d(m, c+3, 3, 0);
            res += x4d(b, c+3, h+3, w+1) * k4d(m, c+3, 3, 1);
            res += x4d(b, c+3, h+3, w+2) * k4d(m, c+3, 3, 2);
            res += x4d(b, c+3, h+3, w+3) * k4d(m, c+3, 3, 3);
            res += x4d(b, c+3, h+3, w+4) * k4d(m, c+3, 3, 4);
            res += x4d(b, c+3, h+3, w+5) * k4d(m, c+3, 3, 5);
            res += x4d(b, c+3, h+3, w+6) * k4d(m, c+3, 3, 6);

            res += x4d(b, c+3, h+4, w) * k4d(m, c+3, 4, 0);
            res += x4d(b, c+3, h+4, w+1) * k4d(m, c+3, 4, 1);
            res += x4d(b, c+3, h+4, w+2) * k4d(m, c+3, 4, 2);
            res += x4d(b, c+3, h+4, w+3) * k4d(m, c+3, 4, 3);
            res += x4d(b, c+3, h+4, w+4) * k4d(m, c+3, 4, 4);
            res += x4d(b, c+3, h+4, w+5) * k4d(m, c+3, 4, 5);
            res += x4d(b, c+3, h+4, w+6) * k4d(m, c+3, 4, 6);

            res += x4d(b, c+3, h+5, w) * k4d(m, c+3, 5, 0);
            res += x4d(b, c+3, h+5, w+1) * k4d(m, c+3, 5, 1);
            res += x4d(b, c+3, h+5, w+2) * k4d(m, c+3, 5, 2);
            res += x4d(b, c+3, h+5, w+3) * k4d(m, c+3, 5, 3);
            res += x4d(b, c+3, h+5, w+4) * k4d(m, c+3, 5, 4);
            res += x4d(b, c+3, h+5, w+5) * k4d(m, c+3, 5, 5);
            res += x4d(b, c+3, h+5, w+6) * k4d(m, c+3, 5, 6);
            
            res += x4d(b, c+3, h+6, w) * k4d(m, c+3, 6, 0);
            res += x4d(b, c+3, h+6, w+1) * k4d(m, c+3, 6, 1);
            res += x4d(b, c+3, h+6, w+2) * k4d(m, c+3, 6, 2);
            res += x4d(b, c+3, h+6, w+3) * k4d(m, c+3, 6, 3);
            res += x4d(b, c+3, h+6, w+4) * k4d(m, c+3, 6, 4);
            res += x4d(b, c+3, h+6, w+5) * k4d(m, c+3, 6, 5);
            res += x4d(b, c+3, h+6, w+6) * k4d(m, c+3, 6, 6);
        } 
        
        y4d(b, m, h, w) = res;
    }

#undef y4d
#undef x4d
#undef k4d
}

/* __global__ void forward_kernel7(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // __shared__ float partialSum[_C][_K][_K];
    __shared__ volatile float partialSum[_C * _K * _K];
    
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockIdx.x;
    int m = blockIdx.y;

    int h = blockIdx.z / W_out;
    int w = blockIdx.z % W_out;

    int c = threadIdx.x;
    int p = threadIdx.y;
    int q = threadIdx.z;

    int tid = c * K * K + p * K + q;

    //partialSum[c][p][q] = x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
    partialSum[tid] = x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
    __syncthreads();

    for (int s=C*K*K/2; s>32; s/=2)
    {
        if (tid < s)
            partialSum[tid] += partialSum[tid+s];
        __syncthreads();
    }

    if (tid < 32)
    {
        partialSum[tid] += partialSum[tid+32];
        partialSum[tid] += partialSum[tid+16];
        partialSum[tid] += partialSum[tid+8];
        partialSum[tid] += partialSum[tid+4];
        partialSum[tid] += partialSum[tid+2];
        partialSum[tid] += partialSum[tid+1];
    }
    
    if (tid == 0)
        y4d(b, m, h, w) = partialSum[0];
    
#undef y4d
#undef x4d
#undef k4d
} */

__global__ void forward_kernel8_C1(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int y4d_ref1 = M * H_out * W_out;
    const int y4d_ref2 = H_out * W_out;
    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define y4d(i3, i2, i1, i0) y[(i3) * (y4d_ref1) + (i2) * (y4d_ref2) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (x4d_ref1) + (i2) * (x4d_ref2) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (k4d_ref1) + (i2) * (k4d_ref2) + (i1) * (K) + i0]
/* #define y4d(i3, i2, i1, i0) y[__mul24(i3, __mul24(M, __mul24(H_out, W_out))) + __mul24(i2, __mul24(H_out, W_out)) + __mul24(i1, W_out) + i0]
#define x4d(i3, i2, i1, i0) x[__mul24(i3, __mul24(C, __mul24(H, W))) + __mul24(i2, __mul24(H, W)) + __mul24(i1, W) + i0]
#define k4d(i3, i2, i1, i0) filter[__mul24(i3, __mul24(C, __mul24(K, K))) + __mul24(i2, __mul24(K, K)) + __mul24(i1, K) + i0] */

    int b = blockIdx.x;
    int m = blockIdx.y;

    /* int tmp = (W_out + BLOCK_SIZE_C1-1) / BLOCK_SIZE_C1;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE_C1 + threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE_C1 + threadIdx.x; */
    int h = blockIdx.z;
    int w = threadIdx.x;

    //if (h < H_out && w < W_out)
    {
        float res = 0.0f;
        int c = 0;
        res += x4d(b, c, h, w) * k4d(m, c, 0, 0);
        res += x4d(b, c, h, w+1) * k4d(m, c, 0, 1);
        res += x4d(b, c, h, w+2) * k4d(m, c, 0, 2);
        res += x4d(b, c, h, w+3) * k4d(m, c, 0, 3);
        res += x4d(b, c, h, w+4) * k4d(m, c, 0, 4);
        res += x4d(b, c, h, w+5) * k4d(m, c, 0, 5);
        res += x4d(b, c, h, w+6) * k4d(m, c, 0, 6);

        res += x4d(b, c, h+1, w) * k4d(m, c, 1, 0);
        res += x4d(b, c, h+1, w+1) * k4d(m, c, 1, 1);
        res += x4d(b, c, h+1, w+2) * k4d(m, c, 1, 2);
        res += x4d(b, c, h+1, w+3) * k4d(m, c, 1, 3);
        res += x4d(b, c, h+1, w+4) * k4d(m, c, 1, 4);
        res += x4d(b, c, h+1, w+5) * k4d(m, c, 1, 5);
        res += x4d(b, c, h+1, w+6) * k4d(m, c, 1, 6);

        res += x4d(b, c, h+2, w) * k4d(m, c, 2, 0);
        res += x4d(b, c, h+2, w+1) * k4d(m, c, 2, 1);
        res += x4d(b, c, h+2, w+2) * k4d(m, c, 2, 2);
        res += x4d(b, c, h+2, w+3) * k4d(m, c, 2, 3);
        res += x4d(b, c, h+2, w+4) * k4d(m, c, 2, 4);
        res += x4d(b, c, h+2, w+5) * k4d(m, c, 2, 5);
        res += x4d(b, c, h+2, w+6) * k4d(m, c, 2, 6);

        res += x4d(b, c, h+3, w) * k4d(m, c, 3, 0);
        res += x4d(b, c, h+3, w+1) * k4d(m, c, 3, 1);
        res += x4d(b, c, h+3, w+2) * k4d(m, c, 3, 2);
        res += x4d(b, c, h+3, w+3) * k4d(m, c, 3, 3);
        res += x4d(b, c, h+3, w+4) * k4d(m, c, 3, 4);
        res += x4d(b, c, h+3, w+5) * k4d(m, c, 3, 5);
        res += x4d(b, c, h+3, w+6) * k4d(m, c, 3, 6);

        res += x4d(b, c, h+4, w) * k4d(m, c, 4, 0);
        res += x4d(b, c, h+4, w+1) * k4d(m, c, 4, 1);
        res += x4d(b, c, h+4, w+2) * k4d(m, c, 4, 2);
        res += x4d(b, c, h+4, w+3) * k4d(m, c, 4, 3);
        res += x4d(b, c, h+4, w+4) * k4d(m, c, 4, 4);
        res += x4d(b, c, h+4, w+5) * k4d(m, c, 4, 5);
        res += x4d(b, c, h+4, w+6) * k4d(m, c, 4, 6);

        res += x4d(b, c, h+5, w) * k4d(m, c, 5, 0);
        res += x4d(b, c, h+5, w+1) * k4d(m, c, 5, 1);
        res += x4d(b, c, h+5, w+2) * k4d(m, c, 5, 2);
        res += x4d(b, c, h+5, w+3) * k4d(m, c, 5, 3);
        res += x4d(b, c, h+5, w+4) * k4d(m, c, 5, 4);
        res += x4d(b, c, h+5, w+5) * k4d(m, c, 5, 5);
        res += x4d(b, c, h+5, w+6) * k4d(m, c, 5, 6);

        res += x4d(b, c, h+6, w) * k4d(m, c, 6, 0);
        res += x4d(b, c, h+6, w+1) * k4d(m, c, 6, 1);
        res += x4d(b, c, h+6, w+2) * k4d(m, c, 6, 2);
        res += x4d(b, c, h+6, w+3) * k4d(m, c, 6, 3);
        res += x4d(b, c, h+6, w+4) * k4d(m, c, 6, 4);
        res += x4d(b, c, h+6, w+5) * k4d(m, c, 6, 5);
        res += x4d(b, c, h+6, w+6) * k4d(m, c, 6, 6);
        
        y4d(b, m, h, w) = res;
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel8_C12(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int y4d_ref1 = M * H_out * W_out;
    const int y4d_ref2 = H_out * W_out;
    const int x4d_ref1 = C * H * W;
    const int x4d_ref2 = H * W;
    const int k4d_ref1 = C * K * K;
    const int k4d_ref2 = K * K;
    
#define y4d(i3, i2, i1, i0) y[(i3) * (y4d_ref1) + (i2) * (y4d_ref2) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (x4d_ref1) + (i2) * (x4d_ref2) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (k4d_ref1) + (i2) * (k4d_ref2) + (i1) * (K) + i0]
/* #define y4d(i3, i2, i1, i0) y[__mul24(i3, __mul24(M, __mul24(H_out, W_out))) + __mul24(i2, __mul24(H_out, W_out)) + __mul24(i1, W_out) + i0]
#define x4d(i3, i2, i1, i0) x[__mul24(i3, __mul24(C, __mul24(H, W))) + __mul24(i2, __mul24(H, W)) + __mul24(i1, W) + i0]
#define k4d(i3, i2, i1, i0) filter[__mul24(i3, __mul24(C, __mul24(K, K))) + __mul24(i2, __mul24(K, K)) + __mul24(i1, K) + i0] */

    int b = blockIdx.x;
    int m = blockIdx.y;

    /* int tmp = (W_out + BLOCK_SIZE_C12-1) / BLOCK_SIZE_C12;
    int h = ( blockIdx.z / tmp) * BLOCK_SIZE_C12 + threadIdx.y;
    int w = ( blockIdx.z % tmp) * BLOCK_SIZE_C12 + threadIdx.x; */
    int h = blockIdx.z;
    int w = threadIdx.x;
    
    //if (h < H_out && w < W_out)
    {
        float res = 0.0f;
        for (int c = 0; c < C; c+=4) 
        {
            res += x4d(b, c, h, w) * k4d(m, c, 0, 0);
            res += x4d(b, c, h, w+1) * k4d(m, c, 0, 1);
            res += x4d(b, c, h, w+2) * k4d(m, c, 0, 2);
            res += x4d(b, c, h, w+3) * k4d(m, c, 0, 3);
            res += x4d(b, c, h, w+4) * k4d(m, c, 0, 4);
            res += x4d(b, c, h, w+5) * k4d(m, c, 0, 5);
            res += x4d(b, c, h, w+6) * k4d(m, c, 0, 6);

            res += x4d(b, c, h+1, w) * k4d(m, c, 1, 0);
            res += x4d(b, c, h+1, w+1) * k4d(m, c, 1, 1);
            res += x4d(b, c, h+1, w+2) * k4d(m, c, 1, 2);
            res += x4d(b, c, h+1, w+3) * k4d(m, c, 1, 3);
            res += x4d(b, c, h+1, w+4) * k4d(m, c, 1, 4);
            res += x4d(b, c, h+1, w+5) * k4d(m, c, 1, 5);
            res += x4d(b, c, h+1, w+6) * k4d(m, c, 1, 6);

            res += x4d(b, c, h+2, w) * k4d(m, c, 2, 0);
            res += x4d(b, c, h+2, w+1) * k4d(m, c, 2, 1);
            res += x4d(b, c, h+2, w+2) * k4d(m, c, 2, 2);
            res += x4d(b, c, h+2, w+3) * k4d(m, c, 2, 3);
            res += x4d(b, c, h+2, w+4) * k4d(m, c, 2, 4);
            res += x4d(b, c, h+2, w+5) * k4d(m, c, 2, 5);
            res += x4d(b, c, h+2, w+6) * k4d(m, c, 2, 6);

            res += x4d(b, c, h+3, w) * k4d(m, c, 3, 0);
            res += x4d(b, c, h+3, w+1) * k4d(m, c, 3, 1);
            res += x4d(b, c, h+3, w+2) * k4d(m, c, 3, 2);
            res += x4d(b, c, h+3, w+3) * k4d(m, c, 3, 3);
            res += x4d(b, c, h+3, w+4) * k4d(m, c, 3, 4);
            res += x4d(b, c, h+3, w+5) * k4d(m, c, 3, 5);
            res += x4d(b, c, h+3, w+6) * k4d(m, c, 3, 6);

            res += x4d(b, c, h+4, w) * k4d(m, c, 4, 0);
            res += x4d(b, c, h+4, w+1) * k4d(m, c, 4, 1);
            res += x4d(b, c, h+4, w+2) * k4d(m, c, 4, 2);
            res += x4d(b, c, h+4, w+3) * k4d(m, c, 4, 3);
            res += x4d(b, c, h+4, w+4) * k4d(m, c, 4, 4);
            res += x4d(b, c, h+4, w+5) * k4d(m, c, 4, 5);
            res += x4d(b, c, h+4, w+6) * k4d(m, c, 4, 6);

            res += x4d(b, c, h+5, w) * k4d(m, c, 5, 0);
            res += x4d(b, c, h+5, w+1) * k4d(m, c, 5, 1);
            res += x4d(b, c, h+5, w+2) * k4d(m, c, 5, 2);
            res += x4d(b, c, h+5, w+3) * k4d(m, c, 5, 3);
            res += x4d(b, c, h+5, w+4) * k4d(m, c, 5, 4);
            res += x4d(b, c, h+5, w+5) * k4d(m, c, 5, 5);
            res += x4d(b, c, h+5, w+6) * k4d(m, c, 5, 6);

            res += x4d(b, c, h+6, w) * k4d(m, c, 6, 0);
            res += x4d(b, c, h+6, w+1) * k4d(m, c, 6, 1);
            res += x4d(b, c, h+6, w+2) * k4d(m, c, 6, 2);
            res += x4d(b, c, h+6, w+3) * k4d(m, c, 6, 3);
            res += x4d(b, c, h+6, w+4) * k4d(m, c, 6, 4);
            res += x4d(b, c, h+6, w+5) * k4d(m, c, 6, 5);
            res += x4d(b, c, h+6, w+6) * k4d(m, c, 6, 6);

            res += x4d(b, c+1, h, w) * k4d(m, c+1, 0, 0);
            res += x4d(b, c+1, h, w+1) * k4d(m, c+1, 0, 1);
            res += x4d(b, c+1, h, w+2) * k4d(m, c+1, 0, 2);
            res += x4d(b, c+1, h, w+3) * k4d(m, c+1, 0, 3);
            res += x4d(b, c+1, h, w+4) * k4d(m, c+1, 0, 4);
            res += x4d(b, c+1, h, w+5) * k4d(m, c+1, 0, 5);
            res += x4d(b, c+1, h, w+6) * k4d(m, c+1, 0, 6);

            res += x4d(b, c+1, h+1, w) * k4d(m, c+1, 1, 0);
            res += x4d(b, c+1, h+1, w+1) * k4d(m, c+1, 1, 1);
            res += x4d(b, c+1, h+1, w+2) * k4d(m, c+1, 1, 2);
            res += x4d(b, c+1, h+1, w+3) * k4d(m, c+1, 1, 3);
            res += x4d(b, c+1, h+1, w+4) * k4d(m, c+1, 1, 4);
            res += x4d(b, c+1, h+1, w+5) * k4d(m, c+1, 1, 5);
            res += x4d(b, c+1, h+1, w+6) * k4d(m, c+1, 1, 6);

            res += x4d(b, c+1, h+2, w) * k4d(m, c+1, 2, 0);
            res += x4d(b, c+1, h+2, w+1) * k4d(m, c+1, 2, 1);
            res += x4d(b, c+1, h+2, w+2) * k4d(m, c+1, 2, 2);
            res += x4d(b, c+1, h+2, w+3) * k4d(m, c+1, 2, 3);
            res += x4d(b, c+1, h+2, w+4) * k4d(m, c+1, 2, 4);
            res += x4d(b, c+1, h+2, w+5) * k4d(m, c+1, 2, 5);
            res += x4d(b, c+1, h+2, w+6) * k4d(m, c+1, 2, 6);

            res += x4d(b, c+1, h+3, w) * k4d(m, c+1, 3, 0);
            res += x4d(b, c+1, h+3, w+1) * k4d(m, c+1, 3, 1);
            res += x4d(b, c+1, h+3, w+2) * k4d(m, c+1, 3, 2);
            res += x4d(b, c+1, h+3, w+3) * k4d(m, c+1, 3, 3);
            res += x4d(b, c+1, h+3, w+4) * k4d(m, c+1, 3, 4);
            res += x4d(b, c+1, h+3, w+5) * k4d(m, c+1, 3, 5);
            res += x4d(b, c+1, h+3, w+6) * k4d(m, c+1, 3, 6);

            res += x4d(b, c+1, h+4, w) * k4d(m, c+1, 4, 0);
            res += x4d(b, c+1, h+4, w+1) * k4d(m, c+1, 4, 1);
            res += x4d(b, c+1, h+4, w+2) * k4d(m, c+1, 4, 2);
            res += x4d(b, c+1, h+4, w+3) * k4d(m, c+1, 4, 3);
            res += x4d(b, c+1, h+4, w+4) * k4d(m, c+1, 4, 4);
            res += x4d(b, c+1, h+4, w+5) * k4d(m, c+1, 4, 5);
            res += x4d(b, c+1, h+4, w+6) * k4d(m, c+1, 4, 6);

            res += x4d(b, c+1, h+5, w) * k4d(m, c+1, 5, 0);
            res += x4d(b, c+1, h+5, w+1) * k4d(m, c+1, 5, 1);
            res += x4d(b, c+1, h+5, w+2) * k4d(m, c+1, 5, 2);
            res += x4d(b, c+1, h+5, w+3) * k4d(m, c+1, 5, 3);
            res += x4d(b, c+1, h+5, w+4) * k4d(m, c+1, 5, 4);
            res += x4d(b, c+1, h+5, w+5) * k4d(m, c+1, 5, 5);
            res += x4d(b, c+1, h+5, w+6) * k4d(m, c+1, 5, 6);

            res += x4d(b, c+1, h+6, w) * k4d(m, c+1, 6, 0);
            res += x4d(b, c+1, h+6, w+1) * k4d(m, c+1, 6, 1);
            res += x4d(b, c+1, h+6, w+2) * k4d(m, c+1, 6, 2);
            res += x4d(b, c+1, h+6, w+3) * k4d(m, c+1, 6, 3);
            res += x4d(b, c+1, h+6, w+4) * k4d(m, c+1, 6, 4);
            res += x4d(b, c+1, h+6, w+5) * k4d(m, c+1, 6, 5);
            res += x4d(b, c+1, h+6, w+6) * k4d(m, c+1, 6, 6);


            res += x4d(b, c+2, h, w) * k4d(m, c+2, 0, 0);
            res += x4d(b, c+2, h, w+1) * k4d(m, c+2, 0, 1);
            res += x4d(b, c+2, h, w+2) * k4d(m, c+2, 0, 2);
            res += x4d(b, c+2, h, w+3) * k4d(m, c+2, 0, 3);
            res += x4d(b, c+2, h, w+4) * k4d(m, c+2, 0, 4);
            res += x4d(b, c+2, h, w+5) * k4d(m, c+2, 0, 5);
            res += x4d(b, c+2, h, w+6) * k4d(m, c+2, 0, 6);

            res += x4d(b, c+2, h+1, w) * k4d(m, c+2, 1, 0);
            res += x4d(b, c+2, h+1, w+1) * k4d(m, c+2, 1, 1);
            res += x4d(b, c+2, h+1, w+2) * k4d(m, c+2, 1, 2);
            res += x4d(b, c+2, h+1, w+3) * k4d(m, c+2, 1, 3);
            res += x4d(b, c+2, h+1, w+4) * k4d(m, c+2, 1, 4);
            res += x4d(b, c+2, h+1, w+5) * k4d(m, c+2, 1, 5);
            res += x4d(b, c+2, h+1, w+6) * k4d(m, c+2, 1, 6);

            res += x4d(b, c+2, h+2, w) * k4d(m, c+2, 2, 0);
            res += x4d(b, c+2, h+2, w+1) * k4d(m, c+2, 2, 1);
            res += x4d(b, c+2, h+2, w+2) * k4d(m, c+2, 2, 2);
            res += x4d(b, c+2, h+2, w+3) * k4d(m, c+2, 2, 3);
            res += x4d(b, c+2, h+2, w+4) * k4d(m, c+2, 2, 4);
            res += x4d(b, c+2, h+2, w+5) * k4d(m, c+2, 2, 5);
            res += x4d(b, c+2, h+2, w+6) * k4d(m, c+2, 2, 6);

            res += x4d(b, c+2, h+3, w) * k4d(m, c+2, 3, 0);
            res += x4d(b, c+2, h+3, w+1) * k4d(m, c+2, 3, 1);
            res += x4d(b, c+2, h+3, w+2) * k4d(m, c+2, 3, 2);
            res += x4d(b, c+2, h+3, w+3) * k4d(m, c+2, 3, 3);
            res += x4d(b, c+2, h+3, w+4) * k4d(m, c+2, 3, 4);
            res += x4d(b, c+2, h+3, w+5) * k4d(m, c+2, 3, 5);
            res += x4d(b, c+2, h+3, w+6) * k4d(m, c+2, 3, 6);

            res += x4d(b, c+2, h+4, w) * k4d(m, c+2, 4, 0);
            res += x4d(b, c+2, h+4, w+1) * k4d(m, c+2, 4, 1);
            res += x4d(b, c+2, h+4, w+2) * k4d(m, c+2, 4, 2);
            res += x4d(b, c+2, h+4, w+3) * k4d(m, c+2, 4, 3);
            res += x4d(b, c+2, h+4, w+4) * k4d(m, c+2, 4, 4);
            res += x4d(b, c+2, h+4, w+5) * k4d(m, c+2, 4, 5);
            res += x4d(b, c+2, h+4, w+6) * k4d(m, c+2, 4, 6);

            res += x4d(b, c+2, h+5, w) * k4d(m, c+2, 5, 0);
            res += x4d(b, c+2, h+5, w+1) * k4d(m, c+2, 5, 1);
            res += x4d(b, c+2, h+5, w+2) * k4d(m, c+2, 5, 2);
            res += x4d(b, c+2, h+5, w+3) * k4d(m, c+2, 5, 3);
            res += x4d(b, c+2, h+5, w+4) * k4d(m, c+2, 5, 4);
            res += x4d(b, c+2, h+5, w+5) * k4d(m, c+2, 5, 5);
            res += x4d(b, c+2, h+5, w+6) * k4d(m, c+2, 5, 6);
            
            res += x4d(b, c+2, h+6, w) * k4d(m, c+2, 6, 0);
            res += x4d(b, c+2, h+6, w+1) * k4d(m, c+2, 6, 1);
            res += x4d(b, c+2, h+6, w+2) * k4d(m, c+2, 6, 2);
            res += x4d(b, c+2, h+6, w+3) * k4d(m, c+2, 6, 3);
            res += x4d(b, c+2, h+6, w+4) * k4d(m, c+2, 6, 4);
            res += x4d(b, c+2, h+6, w+5) * k4d(m, c+2, 6, 5);
            res += x4d(b, c+2, h+6, w+6) * k4d(m, c+2, 6, 6);

            
            res += x4d(b, c+3, h, w) * k4d(m, c+3, 0, 0);
            res += x4d(b, c+3, h, w+1) * k4d(m, c+3, 0, 1);
            res += x4d(b, c+3, h, w+2) * k4d(m, c+3, 0, 2);
            res += x4d(b, c+3, h, w+3) * k4d(m, c+3, 0, 3);
            res += x4d(b, c+3, h, w+4) * k4d(m, c+3, 0, 4);
            res += x4d(b, c+3, h, w+5) * k4d(m, c+3, 0, 5);
            res += x4d(b, c+3, h, w+6) * k4d(m, c+3, 0, 6);

            res += x4d(b, c+3, h+1, w) * k4d(m, c+3, 1, 0);
            res += x4d(b, c+3, h+1, w+1) * k4d(m, c+3, 1, 1);
            res += x4d(b, c+3, h+1, w+2) * k4d(m, c+3, 1, 2);
            res += x4d(b, c+3, h+1, w+3) * k4d(m, c+3, 1, 3);
            res += x4d(b, c+3, h+1, w+4) * k4d(m, c+3, 1, 4);
            res += x4d(b, c+3, h+1, w+5) * k4d(m, c+3, 1, 5);
            res += x4d(b, c+3, h+1, w+6) * k4d(m, c+3, 1, 6);

            res += x4d(b, c+3, h+2, w) * k4d(m, c+3, 2, 0);
            res += x4d(b, c+3, h+2, w+1) * k4d(m, c+3, 2, 1);
            res += x4d(b, c+3, h+2, w+2) * k4d(m, c+3, 2, 2);
            res += x4d(b, c+3, h+2, w+3) * k4d(m, c+3, 2, 3);
            res += x4d(b, c+3, h+2, w+4) * k4d(m, c+3, 2, 4);
            res += x4d(b, c+3, h+2, w+5) * k4d(m, c+3, 2, 5);
            res += x4d(b, c+3, h+2, w+6) * k4d(m, c+3, 2, 6);

            res += x4d(b, c+3, h+3, w) * k4d(m, c+3, 3, 0);
            res += x4d(b, c+3, h+3, w+1) * k4d(m, c+3, 3, 1);
            res += x4d(b, c+3, h+3, w+2) * k4d(m, c+3, 3, 2);
            res += x4d(b, c+3, h+3, w+3) * k4d(m, c+3, 3, 3);
            res += x4d(b, c+3, h+3, w+4) * k4d(m, c+3, 3, 4);
            res += x4d(b, c+3, h+3, w+5) * k4d(m, c+3, 3, 5);
            res += x4d(b, c+3, h+3, w+6) * k4d(m, c+3, 3, 6);

            res += x4d(b, c+3, h+4, w) * k4d(m, c+3, 4, 0);
            res += x4d(b, c+3, h+4, w+1) * k4d(m, c+3, 4, 1);
            res += x4d(b, c+3, h+4, w+2) * k4d(m, c+3, 4, 2);
            res += x4d(b, c+3, h+4, w+3) * k4d(m, c+3, 4, 3);
            res += x4d(b, c+3, h+4, w+4) * k4d(m, c+3, 4, 4);
            res += x4d(b, c+3, h+4, w+5) * k4d(m, c+3, 4, 5);
            res += x4d(b, c+3, h+4, w+6) * k4d(m, c+3, 4, 6);

            res += x4d(b, c+3, h+5, w) * k4d(m, c+3, 5, 0);
            res += x4d(b, c+3, h+5, w+1) * k4d(m, c+3, 5, 1);
            res += x4d(b, c+3, h+5, w+2) * k4d(m, c+3, 5, 2);
            res += x4d(b, c+3, h+5, w+3) * k4d(m, c+3, 5, 3);
            res += x4d(b, c+3, h+5, w+4) * k4d(m, c+3, 5, 4);
            res += x4d(b, c+3, h+5, w+5) * k4d(m, c+3, 5, 5);
            res += x4d(b, c+3, h+5, w+6) * k4d(m, c+3, 5, 6);
            
            res += x4d(b, c+3, h+6, w) * k4d(m, c+3, 6, 0);
            res += x4d(b, c+3, h+6, w+1) * k4d(m, c+3, 6, 1);
            res += x4d(b, c+3, h+6, w+2) * k4d(m, c+3, 6, 2);
            res += x4d(b, c+3, h+6, w+3) * k4d(m, c+3, 6, 3);
            res += x4d(b, c+3, h+6, w+4) * k4d(m, c+3, 6, 4);
            res += x4d(b, c+3, h+6, w+5) * k4d(m, c+3, 6, 5);
            res += x4d(b, c+3, h+6, w+6) * k4d(m, c+3, 6, 6);
        } 
        
        y4d(b, m, h, w) = res;
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    
    // 7
    /* int W_OUT = W-K+1; 
    int H_OUT = H-K+1; 
    dim3 gridDim(B, M, W_OUT*H_OUT);
    dim3 blockDim(C, K, K); */
    // 6
    /* int W_OUT = (W+K-1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    int H_OUT = (H+K-1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(B, M, W_OUT*H_OUT);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);  */
    // 5 bad
    /* int W_OUT = (W-K+1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    int H_OUT = (H-K+1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(B, M, W_OUT*H_OUT);
    dim3 blockDim(BLOCK_SIZE+K-1, BLOCK_SIZE+K-1, 1); */
    // 3
    /* int W_OUT = (W+K-1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    int H_OUT = (H+K-1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(B, M, W_OUT*H_OUT);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1); */
    // 4 bad
    /* int W_OUT = (W+K-1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    int H_OUT = (H+K-1 + BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(B, M, W_OUT*H_OUT);
    dim3 blockDim(BLOCK_SIZE+K-1, BLOCK_SIZE+K-1, 1); */
    // 0,1,2
    /* dim3 gridDim((B+511)/512);
    dim3 blockDim(512); */

    cudaMemcpyToSymbol(filter, w.dptr_, M * C * K * K * sizeof(float));

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    //forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    // 6
    /* int NUM_STREAM;
    if (C == 1)
        NUM_STREAM = NUM_STREAM_C1;
    else
        NUM_STREAM = NUM_STREAM_C12;

    const int stream_batch_size = B / NUM_STREAM;
    const int stream_input_size = stream_batch_size * C * H * W;
    const int stream_output_size = stream_batch_size * M * (H-K+1) * (W-K+1);

    cudaStream_t* streams = new cudaStream_t[NUM_STREAM];

    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamCreate(&streams[i]);

        const float* x_stream = x.dptr_ + i * stream_input_size;
        float* y_stream = y.dptr_ + i * stream_output_size;
  
        if (C == 1)
        {
            int W_OUT = (W-K+1 + BLOCK_SIZE_C1-1) / BLOCK_SIZE_C1;
            int H_OUT = (H-K+1 + BLOCK_SIZE_C1-1) / BLOCK_SIZE_C1;
            dim3 gridDim(stream_batch_size, M, W_OUT*H_OUT);
            dim3 blockDim(BLOCK_SIZE_C1, BLOCK_SIZE_C1, 1); 
            forward_kernel6_C1<<<gridDim, blockDim, 0, streams[i]>>>(y_stream, x_stream, stream_batch_size, M, C, H, W, K);
        }
        else
        {
            int W_OUT = (W-K+1 + BLOCK_SIZE_C12-1) / BLOCK_SIZE_C12;
            int H_OUT = (H-K+1 + BLOCK_SIZE_C12-1) / BLOCK_SIZE_C12;
            dim3 gridDim(stream_batch_size, M, W_OUT*H_OUT);
            dim3 blockDim(BLOCK_SIZE_C12, BLOCK_SIZE_C12, 1); 
            forward_kernel6_C12<<<gridDim, blockDim, 0, streams[i]>>>(y_stream, x_stream, stream_batch_size, M, C, H, W, K);
        }
    }
    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete [] streams; */
   
    /* if (C == 1)
        forward_kernel5_C1<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
    else
        forward_kernel5_C12<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B, M, C, H, W, K); */
    // 8
    int NUM_STREAM;
    if (C == 1)
        NUM_STREAM = NUM_STREAM_C1;
    else
        NUM_STREAM = NUM_STREAM_C12;

    const int stream_batch_size = B / NUM_STREAM;
    const int stream_input_size = stream_batch_size * C * H * W;
    const int stream_output_size = stream_batch_size * M * (H-K+1) * (W-K+1);

    cudaStream_t* streams = new cudaStream_t[NUM_STREAM];

    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamCreate(&streams[i]);

        const float* x_stream = x.dptr_ + i * stream_input_size;
        float* y_stream = y.dptr_ + i * stream_output_size;
  
        if (C == 1)
        {
            /* int W_OUT = (W-K+1 + BLOCK_SIZE_C1-1) / BLOCK_SIZE_C1;
            int H_OUT = (H-K+1 + BLOCK_SIZE_C1-1) / BLOCK_SIZE_C1; */
            /* dim3 gridDim(stream_batch_size, M, 1);
            dim3 blockDim(W-K+1, H-K+1, 1);  */
            dim3 gridDim(stream_batch_size, M, W-K+1);
            dim3 blockDim(H-K+1); 
            forward_kernel8_C1<<<gridDim, blockDim, 0, streams[i]>>>(y_stream, x_stream, stream_batch_size, M, C, H, W, K);
        }
        else
        {
            /* int W_OUT = (W-K+1 + BLOCK_SIZE_C12-1) / BLOCK_SIZE_C12;
            int H_OUT = (H-K+1 + BLOCK_SIZE_C12-1) / BLOCK_SIZE_C12; */
            /* dim3 gridDim(stream_batch_size, M, 1);
            dim3 blockDim(W-K+1, H-K+1, 1);  */
            dim3 gridDim(stream_batch_size, M, W-K+1);
            dim3 blockDim(H-K+1); 
            forward_kernel8_C12<<<gridDim, blockDim, 0, streams[i]>>>(y_stream, x_stream, stream_batch_size, M, C, H, W, K);
        }
    }
    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete [] streams;
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed for ECE408");
}
}
}

#endif