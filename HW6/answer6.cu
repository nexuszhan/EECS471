// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void castFloat2Char(float* inputImage, unsigned char* ucharImage, int w, int h, int c)
{
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tz = threadIdx.z;
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int bz = blockIdx.z;

  unsigned int x_out = tx + bx * blockDim.x;
  unsigned int y_out = ty + by * blockDim.y;
  unsigned int z_out = tz + bz * blockDim.z;
  
  if (z_out < c && y_out < h && x_out < w)
  {
    int i = z_out*w*h + y_out*w + x_out;
    ucharImage[i] = (unsigned char) (255.0 * inputImage[i]);
  }
}

__global__ void convertRGB2Gray(unsigned char* ucharImage, unsigned char* grayImage, int w, int h)
{
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int x_out = tx + bx * blockDim.x;
  unsigned int y_out = ty + by * blockDim.y;

  if (y_out < h && x_out < w)
  {
    unsigned int i = y_out * w + x_out;
    unsigned char r = ucharImage[3*i];
    unsigned char g = ucharImage[3*i+1];
    unsigned char b = ucharImage[3*i+2];
    grayImage[i] = (unsigned char) (0.21*(float)r + 0.71*(float)g + 0.07*(float)b);
  }
}

__global__ void computeHistogram(unsigned char* grayImage, int* histogram, int w, int h)
{
  __shared__ int hist[HISTOGRAM_LENGTH];

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int x_out = tx + bx * blockDim.x;
  unsigned int y_out = ty + by * blockDim.y;

  unsigned int tmp = tx + ty * blockDim.x;
  if (tmp < HISTOGRAM_LENGTH)
  {
    hist[tmp] = 0;
  }
  __syncthreads();
  
  unsigned int x_stride = blockDim.x * gridDim.x;
  unsigned int y_stride = blockDim.y * gridDim.y;
  while (y_out < h && x_out < w)
  {
    unsigned int i = y_out * w + x_out;
    atomicAdd(&(hist[grayImage[i]]),1);
    x_out += x_stride;
    y_out += y_stride;
  }
  __syncthreads();

  if (tmp < HISTOGRAM_LENGTH)
  {
    atomicAdd(&(histogram[tmp]), hist[tmp]);
  }
}

__global__ void computCDF(int* histogram, float* cdf, int w, int h)
{
  __shared__ float partialScan[HISTOGRAM_LENGTH];
  unsigned int tx = threadIdx.x;
  unsigned int bx = blockIdx.x;
  unsigned int i = tx + bx * HISTOGRAM_LENGTH;

  if (i < HISTOGRAM_LENGTH)
  {
    partialScan[i] = (float)histogram[i] / (float)(w*h);
  }
  __syncthreads();

  for (unsigned int stride=1; stride<=HISTOGRAM_LENGTH; stride*=2)
  {
    unsigned int index = (tx+1)*stride*2 - 1;
    if (index < HISTOGRAM_LENGTH)
    {
      partialScan[index] += partialScan[index-stride];
    }
    __syncthreads();
  }

  for (unsigned int stride=HISTOGRAM_LENGTH/2; stride>0; stride/=2)
  {
    __syncthreads();
    unsigned int index = (tx+1)*stride*2 - 1;
    if (index+stride < HISTOGRAM_LENGTH)
    {
      partialScan[index+stride] += partialScan[index];
    }
  }
  __syncthreads();

  if (i < HISTOGRAM_LENGTH)
  {
    cdf[i] = partialScan[i];
  }
}

__global__ void equalize(float* cdf, unsigned char* ucharImage, int w, int h, int c)
{
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tz = threadIdx.z;
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int bz = blockIdx.z;

  unsigned int x_out = tx + bx * blockDim.x;
  unsigned int y_out = ty + by * blockDim.y;
  unsigned int z_out = tz + bz * blockDim.z;
  
  if (z_out < c && y_out < h && x_out < w)
  {
    int i = z_out*w*h + y_out*w + x_out;
    float x = 255.0 * (cdf[ucharImage[i]]-cdf[0]) / (1.0-cdf[0]);
    if (x < 0.0)
    {
      x = 0.0;
    }
    if (x > 255.0)
    {
      x = 255.0;
    }
    ucharImage[i] = (unsigned char)x;
  }
}

__global__ void cast2Float(unsigned char* ucharImage, float* outputImage, int w, int h, int c)
{
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tz = threadIdx.z;
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int bz = blockIdx.z;

  unsigned int x_out = tx + bx * blockDim.x;
  unsigned int y_out = ty + by * blockDim.y;
  unsigned int z_out = tz + bz * blockDim.z;
  
  if (z_out < c && y_out < h && x_out < w)
  {
    int i = z_out*w*h + y_out*w + x_out;
    outputImage[i] = (float)(ucharImage[i] / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  
  //@@ Insert more code here
  float* deviceInputImageData;
  float* deviceOutputImageData;
  unsigned char* deviceRGBImage;
  unsigned char* deviceGrayImage;
  int* deviceHistogram;
  float* deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int image_size = imageWidth * imageHeight * imageChannels;
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  
  cudaMalloc((void**)& deviceInputImageData, sizeof(float)*image_size);
  cudaMalloc((void**)& deviceOutputImageData, sizeof(float)*image_size);
  cudaMalloc((void**)& deviceRGBImage, sizeof(unsigned char)*image_size);
  cudaMalloc((void**)& deviceGrayImage, sizeof(unsigned char)*imageWidth*imageHeight);
  cudaMalloc((void**)& deviceHistogram, sizeof(int)*HISTOGRAM_LENGTH);
  cudaMemset(deviceHistogram, 0, sizeof(int)*HISTOGRAM_LENGTH);
  cudaMalloc((void**)& deviceCDF, sizeof(float)*HISTOGRAM_LENGTH);

  cudaMemcpy(deviceInputImageData, hostInputImageData, sizeof(float)*image_size, cudaMemcpyHostToDevice);

  dim3 DimGrid1(ceil(double(imageWidth)/16.0), ceil(double(imageHeight)/16.0), ceil(double(imageChannels)/4.0));
  dim3 DimBlock1(16, 16, 4);
  dim3 DimGrid2(ceil(double(imageWidth)/32.0), ceil(double(imageHeight)/32.0), 1);
  dim3 DimBlock2(32, 32, 1);
  dim3 DimBlock3(HISTOGRAM_LENGTH,1,1);
  dim3 DimGrid3(1,1,1);
  
  // Cast the image from float to unsigned char
  castFloat2Char<<<DimGrid1, DimBlock1>>>(deviceInputImageData, deviceRGBImage, imageWidth, imageHeight, imageChannels);

  // Convert image from RGB to GrayScale
  convertRGB2Gray<<<DimGrid2, DimBlock2>>>(deviceRGBImage, deviceGrayImage, imageWidth, imageHeight);

  // Compute histogram of grayImgage
  computeHistogram<<<DimGrid2, DimBlock2>>>(deviceGrayImage, deviceHistogram, imageWidth, imageHeight);

  // Compute Cumulative Distribution Function
  computCDF<<<DimGrid3, DimBlock3>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);

  // Apply equalization function
  equalize<<<DimGrid1, DimBlock1>>>(deviceCDF, deviceRGBImage, imageWidth, imageHeight, imageChannels);

  // Cast back to float
  cast2Float<<<DimGrid1, DimBlock1>>>(deviceRGBImage, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, sizeof(float)*image_size, cudaMemcpyDeviceToHost);
  wbImage_setData(outputImage, hostOutputImageData);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceRGBImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);

  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
