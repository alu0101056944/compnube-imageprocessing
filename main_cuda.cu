/**
 * Author: Marcos Barrios
 * Since: 22/04/2024
 * Description: Print execution time of cuda parallel image processing oil
 *    painting algorithm.
*/

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>

#include <sys/time.h>
#include <time.h>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

namespace fs = std::filesystem;

#include "includes/image_process_cuda.h"

void printExecutionTime(const cv::Mat& image) {
  const int kAmountOfIterations = 5;
  float totalTime = 0.0f;

  // Warm-up run
  getProcessedImageParallelCUDA(image);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  cudaEventRecord(start);
  for (size_t i = 0; i < kAmountOfIterations; ++i) {
    const cv::Mat processedImage = getProcessedImageParallelCUDA(image);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalTime += milliseconds;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << totalTime / kAmountOfIterations;
  std::cout << " miliseconds. (Execution time)" << std::endl;
}

void writeImage(const cv::Mat& image, const fs::path& path) {
  std::cout << "Writing image, please wait..." << std::endl;
  const cv::Mat outputImage = getProcessedImageParallelCUDA(image);
  const std::string kOutputPath = (path.parent_path() / path.stem())
      .string() + "_processed" + path.extension().string();
  cv::imwrite(kOutputPath, outputImage);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: <path to image> (expected at least one argument).";
    std::cout << std::endl;
    return -1;
  }

  const std::string kFilePath = argv[1];
  fs::path inputPath(kFilePath);
  if (!fs::exists(inputPath) ||
      !fs::is_regular_file(inputPath)) {
    std::cout << "Invalid file path." << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(kFilePath, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Error: Unable to load image." << std::endl;
    return -1;
  }

  printExecutionTime(image);
  writeImage(image, inputPath);

  return 0;
}
