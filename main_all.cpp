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

#include <opencv2/opencv.hpp>

#include "includes/image_process_sequential.h"
#include "includes/image_process_parallel.h"

namespace fs = std::filesystem;

void printExecutionTimes(const std::array<cv::Mat, 4>& images) {
  std::cout << "Paralell:" << std::endl;
  std::cout << "Size \t\t T. Exec (Seconds)" << std::endl;

  const int kAmountOfIterations = 5;
  for (const cv::Mat& image : images) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < kAmountOfIterations; ++i) {
      const cv::Mat processedImage = getProcessedImageParallel(image);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    const cv::Mat tempImage = getProcessedImageParallel(image);
    std::cout << tempImage.rows << "x" << tempImage.cols << "\t\t";
    auto timeSpan =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << timeSpan.count() / kAmountOfIterations << std::endl;
  }

  std::cout << "Sequential:" << std::endl;
  std::cout << "Size \t\t T. Exec (Seconds)" << std::endl;

  for (const cv::Mat& image : images) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < kAmountOfIterations; ++i) {
      const cv::Mat processedImage = getProcessedImageSequential(image);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    const cv::Mat tempImage = getProcessedImageSequential(image);
    std::cout << tempImage.rows << "x" << tempImage.cols << "\t\t";
    auto timeSpan =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << timeSpan.count() / kAmountOfIterations << std::endl;
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
  std::array<std::string, 4> paths = { "docs/f1.jpg", "docs/f1_50%.jpg",
    "docs/f1_25%.jpg", "docs/f1_10%.jpg" };
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
