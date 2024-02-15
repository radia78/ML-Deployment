#include <iostream>
#include <torch/torch.h>
#include <cmath>
#include "person_segmentator/person_segmentator.h"

// declaring the namespaces
using namespace cv;
using namespace std;

// fixed dimension
#define HEIGHT 224
#define WIDTH 224
#define FPS 24

int main()
{
    // setup the camera feed
    const string window_name = "Camera feed";
    auto modelFilePath = "/Users/radiakbar/Projects/Object-Segmentation/assets/model.pt";

    // fps variables
    double start_time, current_time;
    double fps;

    // Create segmentator
    PersonSegmentator ps(modelFilePath, "cpu");

    // Open the default video camera and initialize the image
    VideoCapture cap = VideoCapture(0);
    // set the video resolution
    cap.set(CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cap.set(CAP_PROP_FPS, FPS);
    Mat frame, imageMask, maskedImage;
    
    // if you can't open the video camera, then print error
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video camera" << endl;
        cin.get();
    }
    
    // naming the camera feed window
    namedWindow(window_name);

    // Infinite loop on the camera feed, unless we press something
    while (cap.isOpened())
    {

        start_time = static_cast<double>(getTickCount());

        // read-in the camera feed
        cap.read(frame);
        resize(frame, frame, Size(HEIGHT, WIDTH), INTER_LINEAR);

        // perform inference on the camera feed;
        imageMask = ps.Inference(frame);

        // Edit the image
        addWeighted(imageMask, 1, frame, 1, 0, maskedImage);

        // compute the frames per second
        current_time = static_cast<double>(getTickCount());
        fps = 1 / ((current_time - start_time) / getTickFrequency());
        
        // display fps
        cv::putText(maskedImage, "FPS: " + to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        // Display the image
        imshow("Semantic Segmentation Predictions", maskedImage);

        if (waitKey(10) == 27) // wait for 10 ms and the esc key to break loop
        {
            cout << "Esc key is pressed by user. Stopping video" << endl;
            break;
        }
    }

    return 0;
}
