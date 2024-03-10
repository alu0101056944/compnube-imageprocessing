/**
 * Author: Marcos Barrios
 * Since: 26/02/2024
 * Description: A oil paint image processing algorithm.
*/

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

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

  const int kAmountOfIterations = 5;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < kAmountOfIterations; ++i) {
    const cv::Mat processedImage = getProcessedImage(image);
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << time_span.count() / kAmountOfIterations;
  std::cout << " seconds. (Execution time)" << std::endl; 

  const cv::Mat outputImage = getProcessedImage(image);
  const std::string kOutputPath = (inputPath.parent_path() / inputPath.stem())
      .string() + "_processed" + inputPath.extension().string();
  cv::imwrite(kOutputPath, outputImage);

  return 0;
}
