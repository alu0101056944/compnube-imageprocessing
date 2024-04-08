/**
 * Author: Marcos Barrios
 * Since: 01/04/2024
 * Description: A oil paint image processing algorithm in parallell using MPI
*/

#ifndef IMAGE_PROCESS_PARALELL_MPI
#define IMAGE_PROCESS_PARALELL_MPI

#include <vector>
#include <opencv2/opencv.hpp>

[[nodiscard]] std::vector<double> getProcessedImageParallelMPI(const cv::Mat& image);

#endif
