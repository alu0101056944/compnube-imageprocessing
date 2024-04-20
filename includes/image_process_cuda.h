/**
 * Author: Marcos Barrios
 * Since: 20/04/2024
 * Description: CUDA version of the same oil painting algorithm
*/

#ifndef IMAGE_PROCESS_PARALELL_CUDA
#define IMAGE_PROCESS_PARALELL_CUDA

#include <vector>

#include <opencv2/opencv.hpp>

[[nodiscard]] cv::Mat getProcessedImageParallelCUDA(const cv::Mat& image,
    int rank, int size);

// outputData is meant to have three R,G,B doubles per pixel.
__global__ void applyFilterOnPixels(const cv::Mat& image,
    std::vector<double>& outputData, int startPixel, int endPixel);

#endif
