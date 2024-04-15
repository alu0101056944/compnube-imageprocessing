#include "../includes/image_process_mpi.h"

#include <algorithm>
#include <math.h>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "mpi.h"

[[nodiscard]] std::vector<double> getImageChunk(const cv::Mat& image,
    int startPixel, int endPixel) {
  std::vector<double> outputChunk((endPixel - startPixel + 1) * 3); // resize. 3 because 3 channels.

  // I should change the otput structure so that each pixel has the values for its channels
  // consecutively instead of separating the whole output with whole areas per channel.

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

    outputChunk[i * 3] = kRFinal;
    outputChunk[i * 3 + 1] = kGFinal;
    outputChunk[i * 3 + 2] = kBFinal;
  }

  return outputChunk;
}

cv::Mat getProcessedImageParallelMPI(const cv::Mat& image, int rank, int size) {
  const int kSize = image.rows * image.cols;
  const int kChunkSize = kSize / size;

  std::vector<double> imageAsVector(kSize * 3); // because 3 channels

  if (kChunkSize < 1) {
    if (rank < kSize) {
      const int kStartPixel = rank;
      const int kEndPixel = rank;
      std::vector<double> chunk = getImageChunk(image, kStartPixel, kEndPixel);
      for (size_t j = 0; j < size; j++) {
        for (size_t i = 0; i < 3; ++i) {
          MPI_Gather(chunk.data(), 3, MPI_DOUBLE,
              (&imageAsVector.data() + (kStartPixel * 3)) + i, 3, MPI_DOUBLE,
                  j, MPI_COMM_WORLD);
        }
      }
    }
  } else {
    const int kStartPixel = rank * kChunkSize;
    const int kEndPixel = rank * kChunkSize + kChunkSize - 1;
    std::vector<double> chunk = getImageChunk(image, kStartPixel, kEndPixel);
    for (size_t j = 0; j < size; j++) {
      for (size_t i = 0; i < 3; ++i) {
        MPI_Gather(chunk.data(), chunk.size(), MPI_DOUBLE,
            (&imageAsVector.data() + (kStartPixel * 3)) + i, chunk.size(),
                MPI_DOUBLE, j, MPI_COMM_WORLD);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return cv::Mat(3, {2, 2, 2}, cv::CV_64FC3, imageAsVector.data());
}
