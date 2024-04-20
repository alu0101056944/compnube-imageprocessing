#include "../includes/image_process_cuda.h"

#include <algorithm>
#include <math.h>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

__global__ void _getImageChunk(const cv::Mat& image,
    std::vector<double>& pixelData) {
  const int kIntensityLevels = 20;
  const int kRadius = 5;
  const int kGridIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
  const int kBlock = kGridIdx * blockDim.x * blockDim.y;
  const int kThreadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int kPixelIndex = kBlock + kThreadId;

  if (kPixelIndex <= std::min((image.cols * image.rows) - 1)) {
    const int kRow = floor(i / image.cols);
    const int kColumn = i % image.cols;

    int maximumIntensity = -1;

    std::unordered_map<int, int> colorTotalsR;
    std::unordered_map<int, int> colorTotalsG;
    std::unordered_map<int, int> colorTotalsB;
    std::unordered_map<int, int> intensityCount;

    for (int ii = std::max(0, kRow - kRadius);
        ii < std::min(image.rows, kRow + kRadius); ++ii) {
      for (int jj = std::max(0, kColumn - kRadius);
            jj < std::min(image.cols, kColumn + kRadius); ++jj) {
        const cv::Vec3b& pixelNeighbor = image.at<cv::Vec3b>(ii, jj);

        const double kR = pixelNeighbor.val[2];
        const double kG = pixelNeighbor.val[1];
        const double kB = pixelNeighbor.val[0];

        int intensity =
            (((kR + kG + kB) / 3) * kIntensityLevels) / 255.0f;
        if (intensity > 255) {
          intensity = 255;
        }

        auto it = intensityCount.find(intensity);
        if (it != intensityCount.end()) {
          ++intensityCount[intensity];
        } else {
          intensityCount[intensity] = 1;
        }

        if (maximumIntensity == -1 || maximumIntensity < intensity) {
          maximumIntensity = intensity;
        }

        if (colorTotalsR.find(intensity) != colorTotalsR.end()) {
          colorTotalsR[intensity] += kR;
        } else {
          colorTotalsR[intensity] = kR;
        }
        if (colorTotalsG.find(intensity) != colorTotalsG.end()) {
          colorTotalsG[intensity] += kG;
        } else {
          colorTotalsG[intensity] = kG;
        }
        if (colorTotalsB.find(intensity) != colorTotalsB.end()) {
          colorTotalsB[intensity] += kB;
        } else {
          colorTotalsB[intensity] = kR;
        }
      }
    }

    const int kRFinal =
        colorTotalsR[maximumIntensity] / intensityCount[maximumIntensity];
    const int kGFinal =
        colorTotalsG[maximumIntensity] / intensityCount[maximumIntensity];
    const int kBFinal =
        colorTotalsB[maximumIntensity] / intensityCount[maximumIntensity];

    pixelData[kStartPixel * 3] = kBFinal;
    pixelData[kStartPixel * 3 + 1] = kBFinal;
    pixelData[kStartPixel * 3 + 2] = kBFinal;
  }
}

cv::Mat getProcessedImageParallelCUDA(const cv::Mat& image, int rank, int size) {
  const int kSize = image.rows * image.cols;
  const int kChunkSize = kSize / size;

  std::vector<double> pixelData(kSize * 3); // because 3 channels
  cv::Mat newImage(image.rows, image.cols, CV_64FC3, pixelData.data());

  
  <<>>_getImageChunk(image, pixelData);

  // because Mat only points to the data, so need to copy it
  cv::Mat outputImage(image.rows, image.cols);
  newImage.copyTo(outputImage);
  return outputImage;
}
