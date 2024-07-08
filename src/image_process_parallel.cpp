#include "../includes/image_process_parallel.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>

[[nodiscard]] cv::Mat getProcessedImageParallel(const cv::Mat& image) {
  cv::Mat outputImage = image.clone();

  const int kIntensityLevels = 20;
  const int kRadius = 5;

#pragma omp parallel for shared(outputImage, image) collapse(2)
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      cv::Vec3b& outputPixel = outputImage.at<cv::Vec3b>(i, j);

      int maximumIntensity = -1;

      std::unordered_map<int, int> colorTotalsR;
      std::unordered_map<int, int> colorTotalsG;
      std::unordered_map<int, int> colorTotalsB;
      std::unordered_map<int, int> intensityCount;

      for (int ii = std::max(0, i - kRadius);
           ii < std::min(image.rows, i + kRadius); ++ii) {
        for (int jj = std::max(0, j - kRadius);
             jj < std::min(image.cols, j + kRadius); ++jj) {
          const cv::Vec3b& pixelNeighbor = image.at<cv::Vec3b>(ii, jj);

          const double kR = pixelNeighbor.val[2];
          const double kG = pixelNeighbor.val[1];
          const double kB = pixelNeighbor.val[0];

          int intensity = (((kR + kG + kB) / 3) * kIntensityLevels) / 255.0f;
          if (intensity > 255) {
            intensity = 255;
          }

          auto it = intensityCount.find(intensity);
          if (it != intensityCount.end()) {
            ++intensityCount[intensity];
          } else {
            intensityCount[intensity] = 1;
          }

          if (maximumIntensity == -1 || maximumIntensity < intensity) {
            maximumIntensity = intensity;
          }

          if (colorTotalsR.find(intensity) != colorTotalsR.end()) {
            colorTotalsR[intensity] += kR;
          } else {
            colorTotalsR[intensity] = kR;
          }
          if (colorTotalsG.find(intensity) != colorTotalsG.end()) {
            colorTotalsG[intensity] += kG;
          } else {
            colorTotalsG[intensity] = kG;
          }
          if (colorTotalsB.find(intensity) != colorTotalsB.end()) {
            colorTotalsB[intensity] += kB;
          } else {
            colorTotalsB[intensity] = kR;
          }
        }
      }

      const int kRFinal =
          colorTotalsR[maximumIntensity] / intensityCount[maximumIntensity];
      const int kGFinal =
          colorTotalsG[maximumIntensity] / intensityCount[maximumIntensity];
      const int kBFinal =
          colorTotalsB[maximumIntensity] / intensityCount[maximumIntensity];

      outputPixel.val[2] = kRFinal;
      outputPixel.val[1] = kGFinal;
      outputPixel.val[0] = kBFinal;
    }
  }

  cv::Mat imageCopy;
  image.copyTo(imageCopy);

  const int kRow = 0;
  const int kCol = 0;

  cv::Vec3b& pixel = imageCopy.at<cv::Vec3b>(0, 0);
  const double kR = pixel.val[2];
  const double kG = pixel.val[1];
  const double kB = pixel.val[0];

  pixel.val[2] = 255;
  pixel.val[1] = 0;
  pixel.val[0] = 0;

  printf("originalimage: (0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d)\n",
    imageCopy.data[kRow * 3 + kCol * 3],
    imageCopy.data[kRow * 3 + kCol * 3 + 1],
    imageCopy.data[kRow * 3 + kCol * 3 + 2],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 2]
  );

  // imageCopy.data[kRow * 3 + kCol * 3] = 0;
  // imageCopy.data[kRow * 3 + kCol * 3 + 1] = 0;
  // imageCopy.data[kRow * 3 + kCol * 3 + 2] = 255;

  printf("originalimage(changed) (0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d),(0:%d,1:%d,2:%d)\n",
    imageCopy.data[kRow * 3 + kCol * 3],
    imageCopy.data[kRow * 3 + kCol * 3 + 1],
    imageCopy.data[kRow * 3 + kCol * 3 + 2],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[kRow * 3 + (kCol + 1) * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + kCol * 3 + 2],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 1],
    imageCopy.data[(kRow + 1) * 3 + (kCol + 1) * 3 + 2]
  );

  // use step in the code

  // cv::cuda::GpuMat newGpuImage(image.rows, image.cols, CV_8UC3);
  // newGpuImage.upload(imageCopy);

  // const int kThreadAmountPerBlock = 32 * 32;
  // const int kBlockAmount =
  //     ceil(((float)image.rows * (float)image.cols) / kThreadAmountPerBlock);
  // dim3 threadsPerBlock(32, 32);
  // _getImageChunk<<<kBlockAmount, threadsPerBlock>>>(imageCopy.data,
  //     newGpuImage.data, image.rows, image.cols);

  // newGpuImage.download(imageCopy);
  return imageCopy;
}
