/**
 * Author: Marcos Barrios
 * Since: 10/03/2024
 * Description: A oil paint image processing algorithm. Normal sequential
 *    version.
*/

#ifndef IMAGE_PROCESS_SEQUENTIAL
#define IMAGE_PROCESS_SEQUENTIAL

#include <opencv2/opencv.hpp>

[[nodiscard]] cv::Mat getProcessedImageParallel(const cv::Mat& image);

#endif
