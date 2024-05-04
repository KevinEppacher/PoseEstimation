#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ComputerVision.h"

int main()
{
    std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";
    ComputerVision::FeatureExtraction training(trainingImagePath);
    training.computeKeypointsAndDescriptors(false);

    std::string videoPath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4";
    ComputerVision::Video video(videoPath);

    std::string activeSet_XYZ_Path = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet_XYZ.csv";

    std::map<int, cv::Point3f> coordinates = LoadXYZCoordinates(activeSet_XYZ_Path);
    
    // for (const auto& pair : coordinates)
    // {
    //     std::cout << "Index: " << pair.first << ", Koordinaten: " << pair.second << std::endl;
    // }
    

    for (auto frame : video.getFrames())
    {
        ComputerVision::FeatureExtraction validation(frame);
        validation.computeKeypointsAndDescriptors(false);

        cv::Mat outputImage;
        ComputerVision::Matcher matcher(training, validation);

        std::vector<cv::DMatch> matches = matcher.matchFeatures();

        std::vector<cv::DMatch> filteredMatches = matcher.filterMatches(150);

        matcher.drawMatches(filteredMatches, outputImage);

        video.putFpsOnImage(outputImage);

        cv::imshow("Brute Force Matching", outputImage);

        if (cv::waitKey(1000 / video.getFPS()) == 27) // Esc-Taste beendet die Schleife
            break;
    }

    cv::destroyAllWindows();

    return 0;
}
