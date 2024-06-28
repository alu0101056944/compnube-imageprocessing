/**
 * Author: Marcos Barrios
 * Since: 20/04/2024
 * Description: CUDA version of the same oil painting algorithm
*/

#ifndef IMAGE_PROCESS_PARALELL_CUDA
#define IMAGE_PROCESS_PARALELL_CUDA

#include <vector>

#include <opencv2/opencv.hpp>

[[nodiscard]] cv::Mat getProcessedImageParallelCUDA(const cv::Mat& image);

#endif
