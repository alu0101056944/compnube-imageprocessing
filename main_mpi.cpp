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
#include "mpi.h"

#include "includes/image_process_mpi.h"

namespace fs = std::filesystem;

void printExecutionTime(const cv::Mat& image, int rank, int size) {
  const int kAmountOfIterations = 5;

  struct timeval timeInit[1], timeEnd[1];
  gettimeofday(timeInit, NULL);
  for (size_t i = 0; i < kAmountOfIterations; ++i) {
    getProcessedImageParallelMPI(image, rank, size);    
  }
  gettimeofday(timeEnd, NULL);

  double kEjecutionTime = timeEnd->tv_sec - timeInit->tv_sec +
      (timeEnd->tv_usec - timeInit->tv_usec) / 1.0e6;
  std::cout << kEjecutionTime / kAmountOfIterations;
  std::cout << " seconds. (Execution time)" << std::endl;
}

void writeImage(const cv::Mat& image, const fs::path& path, int rank, int size) {
  std::cout << "Writing image, please wait..." << std::endl;
  const cv::Mat outputImage = getProcessedImageParallelMPI(image, rank, size);
  const std::string kOutputPath = (path.parent_path() / path.stem())
      .string() + "_processed" + path.extension().string();
  cv::imwrite(kOutputPath, outputImage);
}

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cout << "Expected 2 arguments, given " << argc << "." << std::endl;
    return -1;
  }

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

  {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i)
          sleep(5);
    }

  printExecutionTime(image, rank, size);
  writeImage(image, inputPath, rank, size);

  MPI_Finalize();
  return 0;
}
