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

__global__ void _getImageChunk(uchar* originalData, uchar* pixelData, int rows, int cols) {
  const int kIntensityLevels = 20;
  const int kRadius = 5;

  const int kPreviousThreadAmount = blockIdx.x * blockDim.x * blockDim.y;
  const int kCurrentBlockThreadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  const int kPixelIndex = kPreviousThreadAmount + kCurrentBlockThreadIndex;

  if (kPixelIndex < rows * cols) {
    const int kRow = floorf(kPixelIndex / cols);
    const int kColumn = kPixelIndex % cols;

    pixelData[kRow * 3 * cols + kColumn * 3 + 2] = 55;
    pixelData[kRow * 3 * cols + kColumn * 3 + 1] = 55;
    pixelData[kRow * 3 * cols + kColumn * 3] = 55;

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
  }
}

cv::Mat getProcessedImageParallelCUDA(const cv::Mat& image) {
  cv::Mat imageCopy;
  image.copyTo(imageCopy);

  const int kRow = 0;
  const int kCol = 0;

  cv::Vec3b& pixel = imageCopy.at<cv::Vec3b>(0, 0);
  const double kR = pixel.val[2];
  const double kG = pixel.val[1];
  const double kB = pixel.val[0];

  pixel.val[2] = 255;
  pixel.val[1] = 0;
  pixel.val[0] = 0;

  printf("originalimage: (0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d)\n",
    imageCopy.data[kRow * 3 + kCol * 3],
    imageCopy.data[kRow * 3 + kCol * 3 + 1],
    imageCopy.data[kRow * 3 + kCol * 3 + 2],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 2]
  );

  // imageCopy.data[kRow * 3 + kCol * 3] = 0;
  // imageCopy.data[kRow * 3 + kCol * 3 + 1] = 0;
  // imageCopy.data[kRow * 3 + kCol * 3 + 2] = 255;

  printf("originalimage(changed) (0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d)\n",
    imageCopy.data[kRow * 3 + kCol * 3],
    imageCopy.data[kRow * 3 + kCol * 3 + 1],
    imageCopy.data[kRow * 3 + kCol * 3 + 2],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 2]
  );

  // use step in the code

  // cv::cuda::GpuMat newGpuImage(image.rows, image.cols, CV_8UC3);
  // newGpuImage.upload(imageCopy);

  // const int kThreadAmountPerBlock = 32 * 32;
  // const int kBlockAmount =
  //     ceil(((float)image.rows * (float)image.cols) / kThreadAmountPerBlock);
  // dim3 threadsPerBlock(32, 32);
  // _getImageChunk<<<kBlockAmount, threadsPerBlock>>>(imageCopy.data,
  //     newGpuImage.data, image.rows, image.cols);

  // newGpuImage.download(imageCopy);
  return imageCopy;
}
