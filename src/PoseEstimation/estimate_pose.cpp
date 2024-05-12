#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ComputerVision.h"

int main(int argc, char **argv)
{
    std::string mode;
    if (argc >= 2)
    {
        mode = argv[1];
    }
    else
    {
        mode = "DefaultMode";
    }

    std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";
    std::string pathToTrainingKeypoints = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet.csv";
    std::string pathToMeasuredKeypoints = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet_XYZ.csv";
    std::string videoPath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4";
    std::string activeSet_XYZ_Path = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet_XYZ.csv";
    std::string imageDirectory = "/home/fhtw_user/catkin_ws/src/HW_3/src/Calibrate/Calibration_Images";

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Could not open camera" << std::endl;
        return -1;
    }

    ComputerVision::FeatureExtraction training(trainingImagePath);

    training.computeKeypointsAndDescriptors(false);

    std::vector<cv::KeyPoint> keypoints1 = training.getKeypoints();

    std::vector<int> filteredKeypointsAndDescriptor = training.filterKeypointsAndDescriptor(activeSet_XYZ_Path);

    training.filterKeypointsAndDescriptor(filteredKeypointsAndDescriptor);

    ComputerVision::Video video(videoPath);

    std::vector<cv::Point3f> worldPoints = LoadXYZCoordinates(activeSet_XYZ_Path);

    cv::Size boardSize(8, 6);

    ComputerVision::CameraCalibrator camera(boardSize);

    camera.loadImagesAndAddChessboardPoints(imageDirectory, false);

    camera.calibrate();

    ComputerVision::PoseEstimator poseEstimator(camera);

    cv::Mat undistortedFrame;
    cv::Mat outputImage;
    cv::Mat rvec, tvec;
    cv::Mat cameraFrame;

    if (mode == "Live")
    {
        while (true)
        {
            cap >> cameraFrame;

            cv::undistort(cameraFrame, undistortedFrame, camera.getCameraMatrix(), camera.getDistCoeffs());

            ComputerVision::FeatureExtraction validation(undistortedFrame);

            validation.computeKeypointsAndDescriptors(false);

            ComputerVision::Matcher matcher(training, validation);

            std::vector<cv::DMatch> matches = matcher.matchFeatures();

            std::vector<cv::DMatch> filteredMatches = matcher.filterMatches(230);

            matcher.drawMatches(filteredMatches, outputImage);

            video.putFpsOnImage(outputImage);

            std::vector<cv::KeyPoint> keypoints = validation.getKeypoints();
            
            if (poseEstimator.estimatePose(filteredMatches, validation.getKeypoints(), worldPoints))
            {
                // poseEstimator.drawCoordinateSystem(cameraFrame);
            }

            cv::imshow("Brute Force Matching", outputImage);
            // cv::imshow("Schauma moi", cameraFrame);

            if (cv::waitKey(1000 / video.getFPS()) == 27)
                break;
        }
    }
    else
    {
        for (auto frame : video.getFrames())
        {
            cv::undistort(frame, undistortedFrame, camera.getCameraMatrix(), camera.getDistCoeffs());

            ComputerVision::FeatureExtraction validation(undistortedFrame);

            validation.computeKeypointsAndDescriptors(false);

            ComputerVision::Matcher matcher(training, validation);

            std::vector<cv::DMatch> matches = matcher.matchFeatures();

            std::vector<cv::DMatch> filteredMatches = matcher.filterMatches(150);

            matcher.drawMatches(filteredMatches, outputImage);

            video.putFpsOnImage(outputImage);

            std::vector<cv::KeyPoint> keypoints = validation.getKeypoints();
            
            if (poseEstimator.estimatePose(filteredMatches, validation.getKeypoints(), worldPoints))
            {
                // poseEstimator.drawCoordinateSystem(frame);
            }

            cv::imshow("Brute Force Matching", outputImage);
            // cv::imshow("Schauma moi", frame);

            if (cv::waitKey(1000 / video.getFPS()) == 27)
                break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
