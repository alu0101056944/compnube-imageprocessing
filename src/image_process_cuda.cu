#include "../includes/image_process_cuda.h"

#include <iostream>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

// __global__ void _getImageChunk(uchar* originalData, uchar* pixelData,
//     const int rows, const int cols) {
//   const int kIntensityLevels = 20;
//   const int kRadius = 5;
//   const int kGridIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
//   const int kBlock = kGridIdx * blockDim.x * blockDim.y;
//   const int kThreadId = threadIdx.y * blockDim.x + threadIdx.x;
//   const int kPixelIndex = kBlock + kThreadId;

//   if (kPixelIndex < rows * cols) {
//     const int kRow = floorf(kPixelIndex / cols);
//     const int kColumn = kPixelIndex % cols;

//     int maximumIntensity = -1;

//     // 255 because there won't be more than 255 intensities
//     int colorTotalsR[255] = {0};
//     int colorTotalsG[255] = {0};
//     int colorTotalsB[255] = {0};
//     int intensityCount[255] = {0};

//     for (int ii = max(0, kRow - kRadius);
//         ii < min(rows, kRow + kRadius); ++ii) {
//       for (int jj = max(0, kColumn - kRadius);
//             jj < min(cols, kColumn + kRadius); ++jj) {
//         const int kRow2 = floorf(ii / cols);
//         const int kColumn2 = ii % cols;
          
//         const double kR = originalData[kRow2 * cols + kColumn2 + 2];
//         const double kG = originalData[kRow2 * cols + kColumn2 + 1];
//         const double kB = originalData[kRow2 * cols + kColumn2];

//         int intensity = (((kR + kG + kB) / 3) * kIntensityLevels) / 255.0f;
//         if (intensity > 255) {
//           intensity = 255;
//         }

//         ++intensityCount[intensity];

//         if (maximumIntensity == -1 || maximumIntensity < intensity) {
//           maximumIntensity = intensity;
//         }

//         colorTotalsR[intensity] += kR;
//         colorTotalsG[intensity] += kR;
//         colorTotalsB[intensity] += kR;
//       }
//     }

//     const int kRFinal =
//         colorTotalsR[maximumIntensity] / intensityCount[maximumIntensity];
//     const int kGFinal =
//         colorTotalsG[maximumIntensity] / intensityCount[maximumIntensity];
//     const int kBFinal =
//         colorTotalsB[maximumIntensity] / intensityCount[maximumIntensity];

//     pixelData[kRow * 3 * cols + kColumn * 3 + 2] = kRFinal;
//     pixelData[kRow * 3 * cols + kColumn * 3 + 1] = kGFinal;
//     pixelData[kRow * 3 * cols + kColumn * 3] = kBFinal;
//   }
// }

cv::Mat getProcessedImageParallelCUDA(const cv::Mat& image) {
  // const int kSize = image.rows * image.cols;
  // std::vector<uchar> pixelData(kSize * 3); // because 3 channels
  // cv::cuda::GpuMat newGpuImage(image.rows, image.cols, CV_8UC3,
  //     pixelData.data());
  // cv::cuda::GpuMat gpuImage(image);
  // const int kThreadAmountPerBlock = 32;
  // const int kBlockAmount =
  //     (int)ceil((image.rows * image.cols) / kThreadAmountPerBlock);
  // _getImageChunk<<<kBlockAmount, kThreadAmountPerBlock>>>(gpuImage.data,
  //     newGpuImage.data, image.rows, image.cols);

  cv::Mat outputImage(image.rows, image.cols, CV_8UC3);
  // newGpuImage.download(outputImage);
  return outputImage;
}
