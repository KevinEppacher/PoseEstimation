#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ComputerVision.h"

int main()
{
    std::string videoPath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4";

    cv::VideoCapture cap(videoPath);

    ComputerVision::Video video(videoPath);

    std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";
    ComputerVision::FeatureExtraction training(trainingImagePath);
    training.computeKeypointsAndDescriptors(false);

    for (auto frame : video.getFrames())
    {
        ComputerVision::FeatureExtraction validation(frame);
        validation.computeKeypointsAndDescriptors(false);

        cv::Mat outputImage;
        ComputerVision::computeBruteForceMatching(training, validation, outputImage, 10);
        
        video.putFpsOnImage(outputImage);

        cv::imshow("Brute Force Matching", outputImage);

        if (cv::waitKey(1000 / video.getFPS()) == 27) // Esc-Taste beendet die Schleife
            break;
    }

    cv::destroyAllWindows();

    return 0;
}
