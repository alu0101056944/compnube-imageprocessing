#include "../includes/image_process_cuda.h"

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda.hpp>
#include <iostream>

__global__ void _getImageChunk(uchar* originalData, uchar* pixelData, int rows, int cols,
    int step) {
  const int kIntensityLevels = 20;
  const int kRadius = 5;

  const int kPreviousThreadAmount = blockIdx.x * blockDim.x * blockDim.y;
  const int kCurrentBlockThreadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  const int kPixelIndex = kPreviousThreadAmount + kCurrentBlockThreadIndex;

  if (kPixelIndex < rows * cols) {
    const int kColumn = blockIdx.x * blockDim.x + threadIdx.x;
    const int kRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (kColumn >= cols || kRow >= rows) return;

    int maximumIntensity = -1;

    // 255 because there won't be more than 255 intensities
    int colorTotalsR[255] = {0};
    int colorTotalsG[255] = {0};
    int colorTotalsB[255] = {0};
    int intensityCount[255] = {0};

    for (int ii = max(0, kRow - kRadius);
        ii < min(rows, kRow + kRadius); ++ii) {
      for (int jj = max(0, kColumn - kRadius);
            jj < min(cols, kColumn + kRadius); ++jj) {
        const double kR = originalData[ii * step + jj * 3 + 2];
        const double kG = originalData[ii * step + jj * 3 + 1];
        const double kB = originalData[ii * step + jj * 3];

        int intensity = (((kR + kG + kB) / 3) * kIntensityLevels) / 255.0f;
        if (intensity > 255) {
          intensity = 255;
        }

        ++intensityCount[intensity];

        if (maximumIntensity == -1 || maximumIntensity < intensity) {
          maximumIntensity = intensity;
        }

        colorTotalsR[intensity] += kR;
        colorTotalsG[intensity] += kG;
        colorTotalsB[intensity] += kB;
      }
    }

    const int kRFinal =
        colorTotalsR[maximumIntensity] / intensityCount[maximumIntensity];
    const int kGFinal =
        colorTotalsG[maximumIntensity] / intensityCount[maximumIntensity];
    const int kBFinal =
        colorTotalsB[maximumIntensity] / intensityCount[maximumIntensity];

    pixelData[kRow * step + kColumn * 3 + 2] = kRFinal;
    pixelData[kRow * step + kColumn * 3 + 1] = kGFinal;
    pixelData[kRow * step + kColumn * 3] = kBFinal;
  }
}

cv::Mat getProcessedImageParallelCUDA(const cv::Mat& image) {
  cv::Mat imageCopy;
  image.copyTo(imageCopy);
  cv::cuda::GpuMat originalImageInGPU(image.rows, image.cols, CV_8UC3);
  originalImageInGPU.upload(imageCopy);

  // use step in the code
  cv::cuda::GpuMat newGpuImage(image.rows, image.cols, CV_8UC3);
  newGpuImage.upload(imageCopy);

  dim3 threadsPerBlock(32, 32);
  dim3 gridSize((image.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, 
          (image.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  _getImageChunk<<<gridSize, threadsPerBlock>>>(originalImageInGPU.data,
      newGpuImage.data, image.rows, image.cols, newGpuImage.step);

  newGpuImage.download(imageCopy);
  return imageCopy;
}
