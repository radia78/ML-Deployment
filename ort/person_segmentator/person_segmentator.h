#ifndef PERSON_SEGMENTATOR_H_
#define PERSON_SEGMENTATOR_H_

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>

// Header for onnxruntime
#include <onnxruntime_cxx_api.h>

template <typename T>
size_t vectorProduct(const std::vector<T>& v) 
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

class PersonSegmentator 
{
    public:
        PersonSegmentator(const std::string& modelFilepath);
        cv::Mat Inference(const cv::Mat& frame);

    private:
        // ORT Environment
        std::shared_ptr<Ort::Env> mEnv;

        // Session
        std::shared_ptr<Ort::Session> mSession;

        // Inputs
        char* mInputName;
        std::vector<int64_t> mInputDims;

        // Outputs
        char* mOutputName;
        std::vector<int64_t> mOutputDims;

        // put the values from the image into the array
        void CreateTensorFromImage(const cv::Mat& img, std::vector<float>& inputTensorValues);

        // convert from tensor to image
        cv::Mat CreateImageFromTensor(const float* floatarr);

        // is greater than function
        bool isGreaterThanThreshold(float value);

};

#endif  // IMAGE_CLASSIFIER_H_