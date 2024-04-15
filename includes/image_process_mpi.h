/**
 * Author: Marcos Barrios
 * Since: 01/04/2024
 * Description: A oil paint image processing algorithm in parallell using MPI
*/

#ifndef IMAGE_PROCESS_PARALELL_MPI
#define IMAGE_PROCESS_PARALELL_MPI

#include <vector>
#include <opencv2/opencv.hpp>

[[nodiscard]] cv::Mat getProcessedImageParallelMPI(const cv::Mat& image, int rank, int size);
[[nodiscard]] std::vector<double> getImageChunk(const cv::Mat& image, int startPixel, int endPixel);

#endif
