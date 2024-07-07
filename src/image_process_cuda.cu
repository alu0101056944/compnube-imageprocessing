#include "../includes/image_process_cuda.h"

#include <iostream>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda.hpp>
#include <iostream>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

__global__ void setColor(uchar* image, int width, int height, int channels) {
  // const int kIntensityLevels = 20;
  // const int kRadius = 5;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < width && y < height) {
    int idx = (y * width + x) * channels;
    image[idx] = 255;        // Blue
    image[idx + 1] = 255;    // Green
    image[idx + 2] = 255;  // Red
  }

  // const int kThreadAmount = blockIdx.x * blockDim.x * blockDim.y;
  // const int kThreadId = threadIdx.y * blockDim.x + threadIdx.x;
  // const int kPixelIndex = kThreadAmount + kThreadId;

  // if (kPixelIndex < rows * cols) {
  //   const int kRow = floorf(kPixelIndex / cols);
  //   const int kColumn = kPixelIndex % cols;

    // int maximumIntensity = -1;

    // // 255 because there won't be more than 255 intensities
    // int colorTotalsR[255] = {0};
    // int colorTotalsG[255] = {0};
    // int colorTotalsB[255] = {0};
    // int intensityCount[255] = {0};

    // for (int ii = max(0, kRow - kRadius);
    //     ii < min(rows, kRow + kRadius); ++ii) {
    //   for (int jj = max(0, kColumn - kRadius);
    //         jj < min(cols, kColumn + kRadius); ++jj) {
    //     const int kRow2 = floorf(ii / cols);
    //     const int kColumn2 = ii % cols;
          
    //     const double kR = originalData[kRow2 * cols + kColumn2 + 2];
    //     const double kG = originalData[kRow2 * cols + kColumn2 + 1];
    //     const double kB = originalData[kRow2 * cols + kColumn2];

    //     int intensity = (((kR + kG + kB) / 3) * kIntensityLevels) / 255.0f;
    //     if (intensity > 255) {
    //       intensity = 255;
    //     }

    //     ++intensityCount[intensity];

    //     if (maximumIntensity == -1 || maximumIntensity < intensity) {
    //       maximumIntensity = intensity;
    //     }

    //     colorTotalsR[intensity] += kR;
    //     colorTotalsG[intensity] += kG;
    //     colorTotalsB[intensity] += kB;
    //   }
    // }

    // const int kRFinal =
    //     colorTotalsR[maximumIntensity] / intensityCount[maximumIntensity];
    // const int kGFinal =
    //     colorTotalsG[maximumIntensity] / intensityCount[maximumIntensity];
    // const int kBFinal =
    //     colorTotalsB[maximumIntensity] / intensityCount[maximumIntensity];

    // pixelData[kRow * 3 * cols + kColumn * 3 + 2] = kRFinal;
    // pixelData[kRow * 3 * cols + kColumn * 3 + 1] = kGFinal;
    // pixelData[kRow * 3 * cols + kColumn * 3] = kBFinal;

    // pixelData[kRow * 3 * cols + kColumn * 3 + 2] = 55;
    // pixelData[kRow * 3 * cols + kColumn * 3 + 1] = 55;
    // pixelData[kRow * 3 * cols + kColumn * 3] = 55;
  // }
}

cv::Mat getProcessedImageParallelCUDA(const cv::Mat& input) {
  // cv::Mat imageCopy(image);

  // cv::cuda::GpuMat newGpuImage(image.rows, image.cols, CV_8UC3);
  // newGpuImage.upload(imageCopy);


  // dim3 block(32, 32);
  // dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);

  // const int kThreadAmountPerBlock = 32;
  // const int kBlockAmount =
  //     (int)ceil((image.rows * image.cols) / kThreadAmountPerBlock);
  // _getImageChunk<<<kBlockAmount, kThreadAmountPerBlock>>>(imageCopy.data,
  //     newGpuImage.data, image.rows, image.cols);



  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
      std::cerr << "No CUDA-capable devices found" << std::endl;
      return cv::Mat();
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using GPU: " << prop.name << " with Compute Capability " << prop.major << "." << prop.minor << std::endl;


  cv::cuda::GpuMat d_input;
  d_input.upload(input);
  
  cv::cuda::GpuMat d_output(input.size(), input.type());
  
  dim3 block(32, 32);
  dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
  std::cout << "Input image size: " << input.cols << "x" << input.rows << " channels: " << input.channels() << std::endl;
  std::cout << "Grid dimensions: " << grid.x << "x" << grid.y << std::endl;
  std::cout << "Block dimensions: " << block.x << "x" << block.y << std::endl;

  setColor<<<grid, block>>>(d_output.ptr<uchar>(), input.cols, input.rows, input.channels());
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cv::Mat output;
  d_output.download(output);
  std::cout << "Output image size: " << output.cols << "x" << output.rows << " channels: " << output.channels() << std::endl;
  
  return output;

  // newGpuImage.download(imageCopy);
  // return imageCopy;
}
