/**
 * Author: Marcos Barrios
 * Since: 10/03/2024
 * Description: Print execution time of sequential image processing oil painting
 *    algorithm.
*/

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <array>
#include <vector>
#include <omp.h>

#include <opencv2/opencv.hpp>

#include "includes/image_process_sequential.h"
#include "includes/image_process_parallel.h"

namespace fs = std::filesystem;

void printExecutionTimes(const std::array<cv::Mat, 4>& images) {
  const int kAmountOfIterations = 5;

  std::cout << "Sequential:" << std::endl;
  std::cout << "Size \t\t T. Exec (Seconds)" << std::endl;
  std::vector<double> executionTimesSequential;
  for (const cv::Mat& image : images) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < kAmountOfIterations; ++i) {
      const cv::Mat processedImage = getProcessedImageSequential(image);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    const cv::Mat tempImage = getProcessedImageSequential(image);
    std::cout << tempImage.rows << "x" << tempImage.cols << "\t\t\t";
    auto timeSpan =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << timeSpan.count() / kAmountOfIterations << std::endl;
    executionTimesSequential.push_back(timeSpan.count() / kAmountOfIterations);
  }

  std::cout << "Paralell:" << std::endl;
  std::cout << "Size \t\t\t ThreadAmount \t\t\t Speed_up" << std::endl;
  std::array<int, 4> threadAmounts = {2, 4, 8, 16};
  for (size_t i = 0; i < images.size(); i++) {
    for (int threadAmount : threadAmounts) {
      omp_set_num_threads(threadAmount);

      auto t1 = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < kAmountOfIterations; ++i) {
        const cv::Mat processedImage = getProcessedImageParallel(images[i]);
      }
      auto t2 = std::chrono::high_resolution_clock::now();

      const cv::Mat tempImage = getProcessedImageParallel(images[i]);
      std::cout << tempImage.rows << "x" << tempImage.cols << "\t\t";
      std::cout << threadAmount << "\t\t";
      auto timeSpan =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
      const double kSpeedUp = (timeSpan.count() / kAmountOfIterations) /
          executionTimesSequential[i];
      std::cout << kSpeedUp << std::endl;
    }
  }
}

void writeImages(const std::array<cv::Mat, 4>& images,
    std::array<std::string, 4> paths) {
  for (size_t i = 0; i < images.size(); ++i) {
    const cv::Mat outputImage = getProcessedImageParallel(images[i]);
    fs::path inputPath(paths[i]);
    const std::string kOutputPath =
        (inputPath.parent_path() / inputPath.stem()).string() +
        "_processed" + inputPath.extension().string();
    cv::imwrite(kOutputPath, outputImage);
  }
}

int main() {
  std::array<std::string, 4> paths = { "docs/f1_10%.jpg", "docs/f1_25%.jpg",
    "docs/f1_50%.jpg", "docs/f1.jpg" };
  for (std::string path : paths) {
    fs::path inputPath(path);
    if (!fs::exists(inputPath) || !fs::is_regular_file(inputPath)) {
      std::cout << "Invalid file path." << std::endl;
      return -1;
    }
  }

  std::array<cv::Mat, 4> images;
  for (size_t i = 0; i < images.size(); ++i) {
    images[i] = cv::imread(paths[i]);
    if (images[i].empty()) {
      std::cerr << "Error: Unable to load image." << std::endl;
      return -1;
    }
  }

  printExecutionTimes(images);
  writeImages(images, paths);
  return 0;
}
