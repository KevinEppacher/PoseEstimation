#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ComputerVision.h"

int main(int argc, char **argv)
{
    std::string mode = argv[1];

    // Initialize the camera
    cv::VideoCapture cap(0); // open the default camera, change the index if you have multiple cameras

    if (!cap.isOpened()) // check if we succeeded
    {
        std::cout << "Could not open camera" << std::endl;
        return -1;
    }

    std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";
    ComputerVision::FeatureExtraction training(trainingImagePath);

    std::string pathToTrainingKeypoints = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet.csv";
    std::string pathToMeasuredKeypoints = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet_XYZ.csv";

    training.computeKeypointsAndDescriptors(false);

    std::vector<cv::KeyPoint> keypoints1 = training.getKeypoints();

    std::vector<int> filterKeypointsAndDescriptor = {0, 2, 3, 11, 13, 14, 15, 19, 20, 29, 33};

    training.filterKeypointsAndDescriptor(filterKeypointsAndDescriptor);

    std::string videoPath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4";
    ComputerVision::Video video(videoPath);

    std::string activeSet_XYZ_Path = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet_XYZ.csv";

    std::vector<cv::Point3f> worldPoints = LoadXYZCoordinates(activeSet_XYZ_Path);

    cv::Size boardSize(8, 6);

    ComputerVision::CameraCalibrator camera(boardSize);

    std::string imageDirectory = "/home/fhtw_user/catkin_ws/src/HW_3/src/Calibrate/Calibration_Images";
    camera.loadImagesAndAddChessboardPoints(imageDirectory, false);

    camera.calibrate();

    ComputerVision::PoseEstimator poseEstimator(camera);

    cv::Mat undistortedFrame;

    if (mode == "Live")
    {
        while (true)
        {

            cv::Mat cameraFrame;
            cap >> cameraFrame;

            cv::undistort(cameraFrame, undistortedFrame, camera.getCameraMatrix(), camera.getDistCoeffs());

            ComputerVision::FeatureExtraction validation(undistortedFrame);
            validation.computeKeypointsAndDescriptors(false);

            cv::Mat outputImage;
            ComputerVision::Matcher matcher(training, validation);

            std::vector<cv::DMatch> matches = matcher.matchFeatures();

            std::vector<cv::DMatch> filteredMatches = matcher.filterMatches(250);

            matcher.drawMatches(filteredMatches, outputImage);

            video.putFpsOnImage(outputImage);

            std::vector<cv::KeyPoint> keypoints = validation.getKeypoints();
            std::vector<cv::Point2f> imagePoints;

            for (const auto &kp : keypoints)
            {
                imagePoints.push_back(kp.pt);
            }

            cv::Mat rvec, tvec;
            if (poseEstimator.estimatePose(filteredMatches, training.getKeypoints(), validation.getKeypoints(), worldPoints))
            {
                poseEstimator.drawCoordinateSystem(cameraFrame);
            }

            cv::imshow("Brute Force Matching", outputImage);
            cv::imshow("Schauma moi", cameraFrame);

            if (cv::waitKey(1000 / video.getFPS()) == 27) // Esc-Taste beendet die Schleife
                break;
        }
    }
    else
    {
        for (auto frame : video.getFrames())
        {
            cv::undistort(frame, undistortedFrame, camera.getCameraMatrix(), camera.getDistCoeffs());

            ComputerVision::FeatureExtraction validation(frame);
            validation.computeKeypointsAndDescriptors(false);

            cv::Mat outputImage;
            ComputerVision::Matcher matcher(training, validation);

            std::vector<cv::DMatch> matches = matcher.matchFeatures();

            std::vector<cv::DMatch> filteredMatches = matcher.filterMatches(150);

            matcher.drawMatches(filteredMatches, outputImage);

            video.putFpsOnImage(outputImage);

            std::vector<cv::KeyPoint> keypoints = validation.getKeypoints();
            std::vector<cv::Point2f> imagePoints;

            for (const auto &kp : keypoints)
            {
                imagePoints.push_back(kp.pt);
            }

            cv::Mat rvec, tvec;
            if (poseEstimator.estimatePose(filteredMatches, training.getKeypoints(), validation.getKeypoints(), worldPoints))
            {
                poseEstimator.drawCoordinateSystem(frame);
            }

            cv::imshow("Brute Force Matching", outputImage);
            cv::imshow("Schauma moi", frame);

            if (cv::waitKey(1000 / video.getFPS()) == 27) // Esc-Taste beendet die Schleife
                break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
