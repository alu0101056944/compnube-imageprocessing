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
#include <sys/timeb.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// #include "includes/image_process_cuda.h"

// void printExecutionTime(const cv::Mat& image) {
//   const int kAmountOfIterations = 5;

//   struct _timeb timeInit;
//   struct _timeb timeEnd;
//   _ftime(&timeInit);
//   for (size_t i = 0; i < kAmountOfIterations; ++i) {
//     const cv::Mat processedImage = getProcessedImageParallelCUDA(image);
//   }
//   _ftime(&timeEnd);

//   time_t kEjecutionTime = timeEnd.time - timeInit.time +
//       (timeEnd.millitm - timeInit.millitm) / 1.0e3;
//   std::cout << kEjecutionTime / kAmountOfIterations;
//   std::cout << " seconds. (Execution time)" << std::endl;
// }

// void writeImage(const cv::Mat& image, const fs::path& path) {
//   std::cout << "Writing image, please wait..." << std::endl;
//   const cv::Mat outputImage = getProcessedImageParallelCUDA(image);
//   const std::string kOutputPath = (path.parent_path() / path.stem())
//       .string() + "_processed" + path.extension().string();
//   cv::imwrite(kOutputPath, outputImage);
// }

int main(int argc, char** argv) {
  std::cout << "something here" << std::endl;
  if (argc < 2) {
    std::cout << "Usage: <path to image> (expected at least one argument).";
    std::cout << std::endl;
    return -1;
  }

  std::cout << "something here" << std::endl;

  const std::string kFilePath = argv[1];
  fs::path inputPath(kFilePath);
  if (!fs::exists(inputPath) ||
      !fs::is_regular_file(inputPath)) {
    std::cout << "Invalid file path." << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(kFilePath);
  // if (image.empty()) {
  //   std::cerr << "Error: Unable to load image." << std::endl;
  //   return -1;
  // }

  // printExecutionTime(image);
  // writeImage(image, inputPath);

  return 0;
}
