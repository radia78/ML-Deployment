#include "person_segmentator.h"
#include <algorithm>

PersonSegmentator::PersonSegmentator(const std::string& modelFilePath, const std::string& deviceString)
{
    /**************** Load torchscript model ******************/
    auto device = torch::Device(deviceString);
    auto model = torch::jit::load(modelFilePath, device);
    mModule = std::make_shared<torch::jit::script::Module>(model);
    mModule->eval();
}

cv::Mat PersonSegmentator::Inference(const cv::Mat& frame)
{
    torch::Tensor input, output;
    input = CreateTensorFromImage(frame);
    output = (mModule->forward({input}).toGenericDict().find("out")->value().toTensor().softmax(1)[0][15] >= 0.7).to(torch::kFloat32);
    
    return CreateImageFromTensor(output);
}

torch::Tensor PersonSegmentator::CreateTensorFromImage(const cv::Mat& frame)
{
    cv::Mat imageRGB, scaledImage;
    cv::cvtColor(frame, imageRGB, cv::COLOR_BGR2RGB);
    imageRGB.convertTo(scaledImage, CV_32F, 1.0f / 255.0f);
    torch::Tensor tensorImage = torch::from_blob(frame.data, {frame.rows, frame.cols, 3}, torch::kByte);
    tensorImage = tensorImage.permute({2, 0, 1});
    tensorImage = tensorImage.div(255);
    tensorImage = tensorImage.unsqueeze(0);
    tensorImage = torch::data::transforms::Normalize<torch::Tensor> (mNormMean, mNormStd)(tensorImage);

    return tensorImage;
}

cv::Mat PersonSegmentator::CreateImageFromTensor(const torch::Tensor& output)
{
    // Create output tensor and image
    torch::Tensor outputMask = torch::zeros({output.size(0), output.size(1), 3});

    // Put the data only at the last part
    outputMask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0}, output);

    // Convert the outputMask into an imageMask
    cv::Mat imageMask(outputMask.size(0), outputMask.size(1), CV_32FC3, outputMask.data_ptr<float>());
    imageMask *= 255;
    imageMask.convertTo(imageMask, CV_8UC3);
    // convert to RGB to BGR and mask the current feed with the segmentation
    cv::cvtColor(imageMask, imageMask, cv::COLOR_RGB2BGR);

    return imageMask;
}
