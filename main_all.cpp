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
#include <sys/time.h>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "includes/image_process_sequential.h"
#include "includes/image_process_parallel.h"

namespace fs = std::filesystem;

void printExecutionTimes(const std::array<cv::Mat, 4>& images) {
  const int kAmountOfIterations = 5;

  std::cout << "Sequential:" << std::endl
            << "Size \t Exec.Time (Seconds)" << std::endl;
  std::vector<double> executionTimesSequential;
  for (const cv::Mat& image : images) {
    struct timeval timeInit[1], timeEnd[1];
    gettimeofday(timeInit, NULL);
    for (size_t i = 0; i < kAmountOfIterations; ++i) {
      const cv::Mat processedImage = getProcessedImageSequential(image);
    }
    gettimeofday(timeEnd, NULL);

    double kEjecutionTime = timeEnd->tv_sec - timeInit->tv_sec +
        (timeEnd->tv_usec - timeInit->tv_usec) / 1.0e6;
    const cv::Mat tempImage = getProcessedImageSequential(image);
    std::cout << std::setw(4) << std::left << tempImage.rows << "x"
              << std::setw(6) << std::left << tempImage.cols
              << std::setw(12) << std::left << kEjecutionTime / kAmountOfIterations
              << std::endl;
    executionTimesSequential.push_back(kEjecutionTime / kAmountOfIterations);
  }

  std::cout << "Paralell:" << std::endl
            << "Size \t ThreadAmount \t Exec.Time(Seq)"
            << "\t Exec. Time (Paral) \t Speed_up \t Efficiency" << std::endl;
  std::array<int, 4> threadAmounts = {2, 4, 8, 16};
  for (size_t i = 0; i < images.size(); i++) {
    for (int threadAmount : threadAmounts) {
      omp_set_num_threads(threadAmount);

      struct timeval timeInit[1], timeEnd[1];
      gettimeofday(timeInit, NULL);
      for (size_t j = 0; j < kAmountOfIterations; ++j) {
        const cv::Mat processedImage = getProcessedImageParallel(images[i]);
      }
      gettimeofday(timeEnd, NULL);

      double kEjecutionTime = timeEnd->tv_sec - timeInit->tv_sec +
          (timeEnd->tv_usec - timeInit->tv_usec) / 1.0e6;
      const double kSpeedUp = executionTimesSequential[i] /
          (kEjecutionTime / kAmountOfIterations);
      const int kEfficiency = (kSpeedUp / threadAmount) * 100;

      const cv::Mat tempImage = getProcessedImageParallel(images[i]);
      std::cout << std::setw(4) << std::left << tempImage.rows << "x"
                << std::setw(6) << std::left << tempImage.cols
                << std::setw(14) << std::left << threadAmount
                << std::setw(17) << std::left << executionTimesSequential[i]
                << std::setw(23) << std::left << kEjecutionTime / kAmountOfIterations
                << std::setw(14) << std::left << kSpeedUp
                << kEfficiency << std::endl;
    }
  }
}

void writeImages(const std::array<cv::Mat, 4>& images,
    std::array<std::string, 4> paths) {
  std::cout << "Writing images, please wait..." << std::endl;
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
