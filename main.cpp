/**
 * Author: Marcos Barrios
 * Since: 26/02/2024
 * Description: A oil paint image processing algorithm.
*/

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <string>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  cv::Mat image = cv::imread("docs/f1.jpg");

  if (image.empty()) {
    std::cerr << "Error: Unable to load image." << std::endl;
    return -1;
  }

  const int kIntensityLevels = 6;

  const int kRadius = 5;
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);

      int maximumIntensity = -1;

      std::unordered_map<int, int> colorTotalsR;
      std::unordered_map<int, int> colorTotalsG;
      std::unordered_map<int, int> colorTotalsB;
      std::unordered_map<int, int> intensityCount;

      const int kMinimumX = j - kRadius;
      const int kMinimumY = i - kRadius;
      for (int ii = j - kRadius; ii < std::min(image.rows, j + kRadius); ++ii) {
        for (int jj = j - kRadius; jj < std::min(image.cols, j + kRadius); ++jj) {
          cv::Vec3b& pixelNeighbor = image.at<cv::Vec3b>(ii, jj);

          const double kR = pixelNeighbor[2];
          const double kG = pixelNeighbor[1];
          const double kB = pixelNeighbor[0];

          // do the calculation of how many intensities are there
          const int kIntensity = (((kR + kG + kB) / 3) * kIntensityLevels) / 255.0f;

          auto it = intensityCount.find(kIntensity);
          if (it != intensityCount.end()) {
            ++intensityCount[kIntensity];
          } else {
            intensityCount[kIntensity] = 1;
          }

          if (maximumIntensity == -1 || maximumIntensity < kIntensity) {
            maximumIntensity = kIntensity;
          }

          auto it = colorTotalsR.find(kIntensity);
          if (it != colorTotalsR.end()) {
            colorTotalsR[kIntensity] += kR;
          } else {
            colorTotalsR[kIntensity] = 0;
          }
          auto it2 = colorTotalsG.find(kIntensity);
          if (it2 != colorTotalsG.end()) {
            colorTotalsG[kIntensity] += kG;
          } else {
            colorTotalsG[kIntensity] = 0;
          }
          auto it3 = colorTotalsB.find(kIntensity);
          if (it3 != colorTotalsB.end()) {
            colorTotalsB[kIntensity] += kB;
          } else {
            colorTotalsB[kIntensity] = 0;
          }
        }
      }

      const int kRFinal =
          colorTotalsR[maximumIntensity] / intensityCount[maximumIntensity];
      const int kGFinal =
          colorTotalsG[maximumIntensity] / intensityCount[maximumIntensity];
      const int kBFinal =
          colorTotalsB[maximumIntensity] / intensityCount[maximumIntensity];

      pixel[2] = kRFinal;
      pixel[1] = kGFinal;
      pixel[0] = kBFinal;
    }
  }

  cv::imwrite("docs/f1_processed.jpg", image);

  return 0;
}
