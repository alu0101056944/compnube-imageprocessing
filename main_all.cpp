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

namespace fs = std::filesystem;

void printExecutionTimes(const std::array<cv::Mat, 4>& images) {
  const int kAmountOfIterations = 5;

  for (cv::Mat image : images) {
      auto t1 = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < kAmountOfIterations; ++i) {
        const cv::Mat processedImage = getProcessedImage(image);
      }
      auto t2 = std::chrono::high_resolution_clock::now();

      auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
      std::cout << time_span.count() / kAmountOfIterations;
      std::cout << " seconds. (Execution time)" << std::endl; 

      const cv::Mat outputImage = getProcessedImageParallel(image);
      const std::string kOutputPath = (inputPath.parent_path() / inputPath.stem())
          .string() + "_processed" + inputPath.extension().string();

      cv::imwrite(kOutputPath, outputImage);
  }
}

int main(int argc, char** argv) {
  std::array<std::string, 4> paths = {
    'docs/f1.jpg',
    'docs/f1_50%.jpg',
    'docs/f1_25%.jpg',
    'docs/f1_10%.jpg',
  }
  for (std::string path : paths) {
    fs::path inputPath(path);
    if (!fs::exists(inputPath) || !fs::is_regular_file(inputPath)) {
      std::cout << "Invalid file path." << std::endl;
      return -1;
    }
  }

  std::array<cv::Mat, 4> images;
  for (size_t i = 0; i < images.size; ++i) {
    image[i] = cv::imread(paths[i]);
    if (image[i].empty()) {
      std::cerr << "Error: Unable to load image." << std::endl;
      return -1;
    }
  }

  printExecutionTimes(images);

  return 0;
}
