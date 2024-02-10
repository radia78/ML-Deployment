#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

cv::Mat loadImage(std::string img_path);
std::vector<std::string> loadCategories(const std::string& filepath);
torch::Tensor preprocessImg(cv::Mat img);

#endif