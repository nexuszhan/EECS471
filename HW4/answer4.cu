#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH  3
#define MASK_RADIUS  1
#define TILE_WIDTH  4
#define SHARED_MEM_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float sdata[SHARED_MEM_WIDTH][SHARED_MEM_WIDTH][SHARED_MEM_WIDTH];
  
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  //printf("%d %d %d\n",tz,ty,tx);
  printf("%d %d %d\n",blockIdx.z, blockIdx.y, blockIdx.x);
  int x_out = tx + blockIdx.x * TILE_WIDTH;
  int y_out = ty + blockIdx.y * TILE_WIDTH;
  int z_out = tz + blockIdx.z * TILE_WIDTH;
  //printf("%d %d %d\n",z_out,y_out,x_out);

  int x_in = x_out - MASK_RADIUS;
  int y_in = y_out - MASK_RADIUS;
  int z_in = z_out - MASK_RADIUS;
  //printf("%d %d %d\n",z_in,y_in,x_in);

  if (x_in >= 0 && x_in < x_size && y_in >=0 && y_in < y_size && z_in >= 0 && z_in < z_size)
  {
    sdata[tz][ty][tx] = input[z_in*x_size*y_size + y_in*x_size + x_in];
  }
  else
  {
    sdata[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH)
  {
    float res = 0.0;
    for (int z_mask=0; z_mask<MASK_WIDTH; z_mask++)
    {
      for (int y_mask=0; y_mask<MASK_WIDTH; y_mask++)
      {
        for (int x_mask=0; x_mask<MASK_WIDTH; x_mask++)
        {
          float mask_val = Mc[z_mask*MASK_WIDTH*MASK_WIDTH+y_mask*MASK_WIDTH+x_mask];
          float input_val = sdata[tz+z_mask][ty+y_mask][tx+x_mask];
          res += mask_val * input_val;
        }
      }
    }

    if (z_out < z_size && y_out < y_size && x_out < x_size)
    {
      output[z_out*x_size*y_size + y_out*x_size + x_out] = res;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, (inputLength-3)*sizeof(float));
  cudaMalloc((void**) &deviceOutput, (inputLength-3)*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength*sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  int z_dim = ceil((double)z_size/(double)TILE_WIDTH);
  int y_dim = ceil((double)y_size/(double)TILE_WIDTH);
  int x_dim = ceil((double)x_size/(double)TILE_WIDTH);
  //printf("%d %d %d\n",z_dim,y_dim,x_dim);
  dim3 DimGrid(x_dim, y_dim, z_dim);
  dim3 DimBlock(SHARED_MEM_WIDTH,SHARED_MEM_WIDTH,SHARED_MEM_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput,deviceOutput,z_size,y_size,x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
