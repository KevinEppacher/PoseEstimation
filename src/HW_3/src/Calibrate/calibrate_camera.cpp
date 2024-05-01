#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "ComputerVision.h"

int main()
{
    // Defining the dimensions of checkerboard
    cv::Size boardSize(8, 6);

    // Creating object of CameraCalibrator
    ComputerVision::CameraCalibrator calibrator(boardSize);

    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    // Path of the folder containing checkerboard images
    std::string path = "/home/fhtw_user/catkin_ws/src/HW_3/src/Calibrate/Calibration_Images/*.jpeg";
    cv::glob(path, images);

    // Looping over all the images in the directory
    for (const auto &imagePath : images)
    {
        cv::Mat image = cv::imread(imagePath);
        if (!image.empty())
        {
            if (calibrator.addChessboardPoints(image))
            {
                std::cout << "Chessboard points detected in: " << imagePath << std::endl;
            }
            else
            {
                std::cout << "Chessboard points not detected in: " << imagePath << std::endl;
            }
        }
    }

    // Calibrate the camera
    cv::Mat testImage = cv::imread(images[0]); // Use any image for imageSize
    if (!testImage.empty())
    {
        calibrator.calibrate(testImage);
    }
    else
    {
        std::cerr << "Error: Test image not found." << std::endl;
    }

    return 0;
}