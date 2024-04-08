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

int main(int argc, char** argv) {
  int rank;
  int size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int kSize = 120;
  std::vector<double> fullMatrix(kSize);

  const int kChunkSize = kSize / size;
  if (kChunkSize < 1) {
    const int kImageSize = kSize;
    if (rank < kImageSize) {
      const int kStartPixel = rank;
      const int kEndPixel = rank;
      std::vector<double> chunk =
        getProcessedImageParallelMPI(fullMatrix, kStartPixel, kEndPixel);
      MPI_Gather(&chunk, 1, MPI_DOUBLE, &fullMatrix, 1, MPI_DOUBLE, 0,
          MPI_COMM_WORLD);
    }
  } else {
    const int kStartPixel = rank * kChunkSize;
    const int kEndPixel = rank * kChunkSize + kChunkSize - 1;
    std::vector<double> chunk =
        getProcessedImageParallelMPI(fullMatrix, kStartPixel, kEndPixel);
    MPI_Gather(&chunk, kChunkSize, MPI_DOUBLE, &fullMatrix, kChunkSize,
        MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    for (int i = 0; i < fullMatrix.size(); ++i) {
      std::cout << fullMatrix[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}
