/**
 * Author: Marcos Barrios
 * Since: 26/02/2024
 * Description: A oil paint image processing algorithm.
*/

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: <path to image> (expected at least one argument).";
    std::cout << std::endl;
    return -1;
  }

  const std::string kFilePath = argv[1];
  fs::path inputPath(kFilePath);

  if (!fs::exists(inputPath) || !fs::is_regular_file(inputPath)) {
    std::cout << "Invalid file path." << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(kFilePath);

  if (image.empty()) {
    std::cerr << "Error: Unable to load image." << std::endl;
    return -1;
  }

  cv::Mat outputImage = image.clone();

  const int kIntensityLevels = 40;

  const int kRadius = 3;
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      cv::Vec3b& outputPixel = outputImage.at<cv::Vec3b>(i, j);

      int maximumIntensity = -1;

      std::unordered_map<int, int> colorTotalsR;
      std::unordered_map<int, int> colorTotalsG;
      std::unordered_map<int, int> colorTotalsB;
      std::unordered_map<int, int> intensityCount;

      for (int ii = std::max(0, i - kRadius);
           ii < std::min(image.rows, i + kRadius); ++ii) {
        for (int jj = std::max(0, j - kRadius);
             jj < std::min(image.cols, j + kRadius); ++jj) {
          cv::Vec3b& pixelNeighbor = image.at<cv::Vec3b>(ii, jj);

          const double kR = pixelNeighbor.val[2];
          const double kG = pixelNeighbor.val[1];
          const double kB = pixelNeighbor.val[0];

          // do the calculation of how many intensities are there
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
          std::cout << "";
        }
      }

      const int kRFinal =
          colorTotalsR[maximumIntensity] / intensityCount[maximumIntensity];
      const int kGFinal =
          colorTotalsG[maximumIntensity] / intensityCount[maximumIntensity];
      const int kBFinal =
          colorTotalsB[maximumIntensity] / intensityCount[maximumIntensity];

      outputPixel.val[2] = kRFinal;
      outputPixel.val[1] = kGFinal;
      outputPixel.val[0] = kBFinal;
    }
  }

  const std::string kOutputPath = (inputPath.parent_path() / inputPath.stem())
      .string() + "_processed" + inputPath.extension().string();

  cv::imwrite(kOutputPath, outputImage);

  return 0;
}
