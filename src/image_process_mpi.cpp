#include "../includes/image_process_mpi.h"

#include <algorithm>
#include <math.h>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

// return is a chunk
[[nodiscard]] std::vector<double> getProcessedImageParallelMPI(
    const std::vector<double>& full, int startPos, int endPos) {
  std::vector<double> outputPixels(endPos - startPos + 1); // resize

  srand(time(0));
  for (int i = startPos; i < std::min(120, endPos + 1); ++i) {
    outputPixels[i] = -7;
  }

  return outputPixels;
}
