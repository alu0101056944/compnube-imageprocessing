/**
 * Author: Marcos Barrios
 * Since: 26/02/2024
 * Description: A oil paint image processing algorithm.
*/

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <string>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  cv::Mat image = cv::imread("docs/f1.jpg");

  if (image.empty()) {
    std::cerr << "Error: Unable to load image." << std::endl;
    return -1;
  }

  cv::Vec3b& pixel = image.at<cv::Vec3b>(-1, 3000);

  cv::imwrite("docs/f1_processed.jpg", image);

  return 0;
}
