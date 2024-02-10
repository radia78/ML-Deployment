#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/torch.h>

// cache the normalizing constants
const std::vector<double> NORM_MEAN = {0.485, 0.456, 0.406};
const std::vector<double> NORM_STD = {0.229, 0.224, 0.225};

// function to preprocess the image
torch::Tensor preprocessImg(cv::Mat img)
{
    // convert cv2 image to tensor
    torch::Tensor tensor_img = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    tensor_img = tensor_img.permute({2, 0, 1});
    tensor_img = tensor_img.div(255);
    tensor_img = tensor_img.unsqueeze(0);
    tensor_img = torch::data::transforms::Normalize<torch::Tensor> (NORM_MEAN, NORM_STD)(tensor_img);

    return tensor_img;
}
