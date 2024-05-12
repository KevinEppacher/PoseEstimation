#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "ComputerVision.h"

int main() {
    // Defining the dimensions of the checkerboard
    cv::Size boardSize(8, 6);

    // Creating an instance of CalibrationManager
    ComputerVision::CameraCalibrator camera(boardSize);

    // Load images and add chessboard points
    std::string imageDirectory = "/home/fhtw_user/catkin_ws/src/HW_3/src/Calibrate/Calibration_Images";
    camera.loadImagesAndAddChessboardPoints(imageDirectory, true);

    // Calibrate the camera
    camera.calibrate();

    return 0;
}