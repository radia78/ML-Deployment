#include "person_segmentator.h"

#include <algorithm>

// Constructor
PersonSegmentator::PersonSegmentator(const std::string& modelFilepath) 
{
    /**************** Create ORT environment ******************/
    std::string instanceName{"Person Segmentator inference"};
    mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());

    /**************** Create ORT session ******************/
    // Set up options for session
    Ort::SessionOptions sessionOptions;
    // Sets graph optimization level (Here, enable all possible optimizations)
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Put only 1 thread
    sessionOptions.SetIntraOpNumThreads(1);
    // Create session by loading the onnx model
    mSession = std::make_shared<Ort::Session>(*mEnv, modelFilepath.c_str(), sessionOptions);

    /**************** Create allocator ******************/
    // Allocator is used to get model information
    Ort::AllocatorWithDefaultOptions allocator;

    /**************** Input info ******************/
    // Get the number of input nodes
    size_t numInputNodes = mSession->GetInputCount();

    // Get the name of the input
    // 0 means the first input of the model
    // The example only has one input, so use 0 here
    mInputName = "input";
    std::cout << "Input Name: " << mInputName << std::endl;

    // Get the type of the input
    // 0 means the first input of the model
    Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    // Get the shape of the input
    mInputDims = inputTensorInfo.GetShape();

    /**************** Output info ******************/
    // Get the number of output nodes
    size_t numOutputNodes = mSession->GetOutputCount();

    // Get the name of the output
    // 0 means the first output of the model
    // The example only has one output, so use 0 here
    mOutputName = "output";
    std::cout << "Output Name: " << mOutputName << std::endl;

    // Get the type of the output
    // 0 means the first output of the model
    Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    // Get the shape of the output
    mOutputDims = outputTensorInfo.GetShape();
}

// Perform inference for a given image
cv::Mat PersonSegmentator::Inference(const cv::Mat& frame) {

    /**************** Preprocessing ******************/
    // Create input tensor (including size and value) from the loaded input image
    // Compute the product of all input dimension
    size_t inputTensorSize = vectorProduct(mInputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    // Load the image into the inputTensorValues
    CreateTensorFromImage(frame, inputTensorValues);

    // Assign memory for input tensor
    // inputTensors will be used by the Session Run for inference
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, mInputDims.data(), mInputDims.size()));

    // Create output tensor (including size and value)
    size_t outputTensorSize = vectorProduct(mOutputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    // Assign memory for output tensors
    // outputTensors will be used by the Session Run for inference
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, mOutputDims.data(), mOutputDims.size()));

    /**************** Inference ******************/
    // 1 means number of inputs and outputs
    // InputTensors and OutputTensors, and inputNames and
    // outputNames are used in Session Run
    std::vector<const char*> inputNames{mInputName};
    std::vector<const char*> outputNames{mOutputName};
    mSession->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);

    /**************** Postprocessing the output result ******************/
    // Get the inference result
    float* floatarr = outputTensors.front().GetTensorMutableData<float>();

    return CreateImageFromTensor(floatarr);
}

// Convert a vector of float arrays into an image mask
cv::Mat PersonSegmentator::CreateImageFromTensor(const float* floatarr)
{
    // Create a vector of floats
    size_t inputTensorSize = vectorProduct(mInputDims);
    std::vector<float> imgVec;
    for (size_t i = 0; i < inputTensorSize; i++)
    {
        if (i % 3 == 0)
        {
            // Only append the data for the first channel
            int j = i / 3;
            float value = (floatarr[15 * mInputDims[2] * mInputDims[3] + j]) >= 0.7 ? 1.0f : 0.0f;
            imgVec.emplace_back(value);
        }
        else
        {
            imgVec.emplace_back(0.0);
        }
    }

    // Create the image mask
    cv::Mat imageMask(mInputDims[2], mInputDims[3], CV_32FC3, imgVec.data());
    imageMask *= 255;
    imageMask.convertTo(imageMask, CV_8UC3);
    cv::cvtColor(imageMask, imageMask, cv::COLOR_RGB2BGR);
    return imageMask;
}

// Create a tensor from the input image
void PersonSegmentator::CreateTensorFromImage(const cv::Mat& img, std::vector<float>& inputTensorValues) 
{
    cv::Mat imageRGB, scaledImage, preprocessedImage;

    /******* Preprocessing *******/
    // Scale image pixels from [0 255] to [0, 1]
    cv::cvtColor(img, imageRGB, cv::COLOR_BGR2RGB);
    imageRGB.convertTo(scaledImage, CV_32F, 1.0f / 255.0f);
    // Convert HWC to CHW
    cv::dnn::blobFromImage(scaledImage, preprocessedImage);

    // Assign the input image to the input tensor
    inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
}
