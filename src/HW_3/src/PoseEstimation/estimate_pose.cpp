#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include "ComputerVision.h"

int main()
{

    std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";

    ComputerVision::FeatureExtraction features(trainingImagePath);

    features.createTrackbar();

    features.extractFeaturesAndDescriptors();

    cv::Mat trainingDescriptors = features.getDescriptor();

    std::vector<cv::KeyPoint> trainingKeypoints = features.getKeypoints();



    return 0;
}