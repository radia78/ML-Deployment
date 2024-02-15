#ifndef PERSON_SEGMENTATOR_H_
#define PERSON_SEGMENTATOR_H_

// Include relevant libraries
#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>

class PersonSegmentator
{
    public:
        PersonSegmentator(const std::string& modelFilePath, const std::string& deviceString);
        cv::Mat Inference(const cv::Mat& frame);

    private:
        // The torchscript model
        std::shared_ptr<torch::jit::script::Module> mModule;

        // Normalizing constants
        const std::vector<double> mNormMean = {0.485, 0.456, 0.406};
        const std::vector<double> mNormStd = {0.229, 0.224, 0.225};

        // Utility functions
        torch::Tensor CreateTensorFromImage(const cv::Mat& frame);
        cv::Mat CreateImageFromTensor(const torch::Tensor& output);
};

#endif
