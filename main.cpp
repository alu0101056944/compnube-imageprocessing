/**
 * Author: Marcos Barrios
 * Since: 26/02/2024
 * Description: A oil paint image processing algorithm.
*/

#include <iostream>
#include <algorithm>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  cv::Mat image = cv::imread("docs/f1.jpg");

  if (image.empty()) {
    std::cerr << "Error: Unable to load image." << std::endl;
    return -1;
  }

  const int kRadius = 5;

  for (size_t i = 0; i < image.rows; ++i) {
    cv::Vec3b& pixel = image.at<cv::Vec3b>(y, x);

    for (size_t j = 0; j < image.cols; ++j) {
      const kMinimumX = j - kRadius;
      const kMinimumY = i - kRadius;
      for (size_t ii = j - kRadius; ii < min(image.rows, j + kRadius); ++ii) {
        for (size_t jj = j - kRadius; jj < min(image.cols, j + kRadius); ++jj) {

        }
      }

      // pixel[0] = 255 - pixel[0];
      // pixel[1] = 255 - pixel[1];
      // pixel[2] = 255 - pixel[2];
    }
  }

  cv::imwrite("docs/f1_processed.jpg", image);

  return 0;
}
