#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "utils/utils.h"

// fixed dimension
#define VID_HEIGHT 320
#define VID_WIDTH 320
#define VID_FPS 24
#define MODEL_PATH "/Users/radiakbar/Projects/torch_cpp/asset/seg_model.pt"

// declaring the namespaces
using namespace cv;
using namespace std;

int main()
{
    // declare the variables that's being used in the program
    torch::jit::script::Module module; // initialize the model
    torch::Tensor result;
    torch::Tensor output_mask = torch::zeros({VID_HEIGHT, VID_WIDTH, 3});

    // Load the model
    try
    {
        module = torch::jit::load(MODEL_PATH);
        module.eval();
    }
    catch (const c10::Error& e)
    {
        // If an error occurs during model loading, catch the exception and print the error message
        cerr << "Error loading the model: " << e.what() << endl;
    }

    // Open the default video camera and initialize the image
    VideoCapture cap = VideoCapture(0);
    // set the video resolution
    cap.set(CAP_PROP_FRAME_WIDTH, VID_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, VID_HEIGHT);
    cap.set(CAP_PROP_FPS, VID_FPS);
    Mat frame, image, masked_img;
    
    // if you can't open the video camera, then print error
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video camera" << endl;
        cin.get();
    }   
    // print the resolution of the camera feed
    cout << "Resolution of the video: " << VID_HEIGHT << " x " << VID_WIDTH << endl;
    
    // naming the camera feed window
    string window_name = "My camera feed";
    namedWindow(window_name);
    
    // Infinite loop on the camera feed, unless we press something
    while (cap.isOpened())
    {
        cap.read(frame);

        // perform inference on the camera feed;
        resize(frame, image, Size(VID_HEIGHT, VID_WIDTH), INTER_LINEAR);
        cvtColor(image, image, COLOR_BGR2RGB);
        result = (module.forward({preprocessImg(image)}).toGenericDict().find("out")->value().toTensor().softmax(1)[0][15] > 0.7).to(torch::kFloat32);
        output_mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), -1}, result);

        // Create an OpenCV Mat object from the numpy array
        Mat image_mask(output_mask.size(0), output_mask.size(1), CV_32FC3, output_mask.data_ptr<float>());
        // multiply by 255 and convert the data type to uint8
        image_mask *= 255;
        image_mask.convertTo(image_mask, CV_8UC3);

        cvtColor(image_mask, image_mask, COLOR_RGB2BGR);

        addWeighted(image_mask, 1, image, 1, 0, masked_img);

        // Display the image
        imshow("Semantic Segmentation Predictions", masked_img);

        if (waitKey(10) == 27) // wait for 10 ms and the esc key to break loop
        {
            cout << "Esc key is pressed by user. Stopping video" << endl;
            break;
        }
    }

    return 0;
}
