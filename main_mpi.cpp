/**
 * Author: Marcos Barrios
 * Since: 01/04/2024
 * Description: Print execution time of parallel image processing oil painting
 *    algorithm with MPI.
*/

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <vector>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

#include "includes/image_process_mpi.h"

namespace fs = std::filesystem;

std::vector calculateAndPrint(const cv::Mat& image, int startPixel, int endPixel) {
  const int kAmountOfIterations = 5;

  struct timeval timeInit[1], timeEnd[1];
  gettimeofday(timeInit, NULL);
  for (size_t i = 0; i < kAmountOfIterations; ++i) {
    // WIP HERE
    std::vector<double> processedImage =
        getProcessedImageParallelMPI(image, startPixel, endPixel);
  }
  gettimeofday(timeEnd, NULL);

  double kEjecutionTime = timeEnd->tv_sec - timeInit->tv_sec +
      (timeEnd->tv_usec - timeInit->tv_usec) / 1.0e6;
  std::cout << kEjecutionTime / kAmountOfIterations;
  std::cout << " seconds. (Execution time)" << std::endl;
}

void printExecutionTime(const cv::Mat& image, int rank, int size) {
  const int kChunkSize = (image.cols * image.rows) / size;
  if (kChunkSize < 1) {
    const int kImageSize = (image.rows * image.cols);
    if (rank < kImageSize) {
      const int kStartPixel = rank;
      const int kEndPixel = rank;
      calculateAndPrint(image, kStartPixel, kEndPixel);
    }
  } else {
    const int kStartPixel = rank * kChunkSize;
    const int kEndPixel = rank * kChunkSize + kChunkSize - 1;
    calculateAndPrint(image, kStartPixel, kEndPixel);
  }
}

void writeImage(const cv::Mat& image, const fs::path& path) {
  std::cout << "Writing image, please wait..." << std::endl;
  const cv::Mat outputImage = getProcessedImageParallelMPI(image);
  const std::string kOutputPath = (path.parent_path() / path.stem())
      .string() + "_processed" + path.extension().string();
  cv::imwrite(kOutputPath, outputImage);
}

int main(int argc, char** argv) {
  const std::string kFilePath = argv[argc - 1];
  fs::path inputPath(kFilePath);
  if (!fs::exists(inputPath) || !fs::is_regular_file(inputPath)) {
    std::cout << "Invalid file path " << kFilePath << "." << std::endl;
    return -1;
  }

  int rank;
  int size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cv::Mat image = cv::imread(kFilePath);
  if (image.empty()) {
    std::cerr << "Error: Unable to load image." << std::endl;
    return -1;
  }

  printExecutionTime(image, rank, size);
  writeImage(image, inputPath);

  MPI_Finalize();
  return 0;
}
