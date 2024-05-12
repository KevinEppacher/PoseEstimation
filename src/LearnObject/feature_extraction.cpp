#include "ComputerVision.h" // Assuming the class definition is in this header

int main()
{
    // ComputerVision::FeatureExtraction extractor("/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png");
    
    // Initialize the camera
    cv::VideoCapture cap(0); // open the default camera, change the index if you have multiple cameras

    if(!cap.isOpened())  // check if we succeeded
    {
        std::cout << "Could not open camera" << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat cameraFrame;
        cap >> cameraFrame; // get a new frame from camera
        ComputerVision::FeatureExtraction extractor2(cameraFrame);

        extractor2.computeKeypointsAndDescriptors(true);

    }

    // extractor.computeKeypointsAndDescriptors(true);
    
    return 0;
}