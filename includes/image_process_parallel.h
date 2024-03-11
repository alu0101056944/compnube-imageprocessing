/**
 * Author: Marcos Barrios
 * Since: 10/03/2024
 * Description: A oil paint image processing algorithm in parallell using OpenMP
*/

#ifndef IMAGE_PROCESS_PARALELL
#define IMAGE_PROCESS_PARALELL

#include <opencv2/opencv.hpp>

[[nodiscard]] cv::Mat getProcessedImageParallel(const cv::Mat& image);

#endif
