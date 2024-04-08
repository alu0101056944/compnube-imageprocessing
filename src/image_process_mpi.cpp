#include "../includes/image_process_mpi.h"

#include <algorithm>
#include <math.h>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "mpi.h"

[[nodiscard]] std::vector<double> getProcessedImageParallelMPI(const cv::Mat& image,
    int startPixel, int endPixel) {
  std::vector<double> outputPixels(endPixel - startPixel + 1); // resize

  const int kIntensityLevels = 20;
  const int kRadius = 5;

  for (int i = startPixel; i < std::min(image.cols * image.rows, endPixel); ++i) {
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

    const int kIndexForR = startPixel;
    const int kIndexForG = startPixel + (image.rows * image.cols);
    const int kIndexForB = startPixel + 2 * (image.rows * image.cols);
    outputPixels[kIndexForR] = kRFinal;
    outputPixels[kIndexForG] = kGFinal;
    outputPixels[kIndexForB] = kBFinal;
  }

  return outputPixels;
}
