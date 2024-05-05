#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ComputerVision.h"

bool EstimatePose(const std::vector<cv::DMatch> &matches,
                  const std::vector<cv::KeyPoint> &keypoints1,
                  const std::vector<cv::KeyPoint> &keypoints2,
                  const std::vector<cv::Point3f> &objectPoints,
                  const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                  cv::Mat &rvec, cv::Mat &tvec)
{
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> selectedObjectPoints;

    for (const auto &match : matches)
    {

        // std::cout<<" matches: "<<std::endl<<match.trainIdx<<std::endl;


        // Add 2D image points from keypoints2 (current frame)
        imagePoints.push_back(keypoints2[match.trainIdx].pt);

        // std::cout<<" objectPoints.size(): "<<std::endl<<match.queryIdx<<std::endl;


        // Add corresponding 3D object points
        int index = match.queryIdx;
        if (index < objectPoints.size())
        {
            selectedObjectPoints.push_back(objectPoints[index]);
        }
    }



    // Check if there are enough points to compute a pose
    if (imagePoints.size() < 4 || selectedObjectPoints.size() < 4)
    {
        std::cerr << "Not enough points to estimate pose." << std::endl;
        return false;
    }

    // std::cout<<" selectedObjectPoints: "<<std::endl<<selectedObjectPoints<<std::endl;
    // std::cout<<" imagePoints: "<<std::endl<<imagePoints<<std::endl;

    std::cout<<" selectedObjectPoints.size(): "<<selectedObjectPoints.size()<<std::endl;
    std::cout<<" imagePoints.size() "<<imagePoints.size()<<std::endl;
    std::cout<<std::endl;


    if (selectedObjectPoints.size() == imagePoints.size())
    {
        bool success = cv::solvePnP(selectedObjectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

        if (!success)
        {
            std::cerr << "Pose estimation failed." << std::endl;
            return false;
        }

        std::cout << "Rotation Vector:\n"
                  << rvec << std::endl;
        std::cout << "Translation Vector:\n"
                  << tvec << std::endl;
        
        // std::cout<<" selectedObjectPoints.size() == imagePoints.size(): "<<std::endl<<(selectedObjectPoints.size() == imagePoints.size())<<std::endl;

    }
    

    // if (!success)
    // {
    //     std::cerr << "Pose estimation failed." << std::endl;
    //     return false;
    // }

    // std::cout << "Rotation Vector:\n"
    //           << rvec << std::endl;
    // std::cout << "Translation Vector:\n"
    //           << tvec << std::endl;

    return true;
}

int main()
{
    std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";
    ComputerVision::FeatureExtraction training(trainingImagePath);
    training.computeKeypointsAndDescriptors(false);

    std::string videoPath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4";
    ComputerVision::Video video(videoPath);

    std::string activeSet_XYZ_Path = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet_XYZ.csv";

    // Koordinaten aus der CSV-Datei laden
    std::vector<cv::Point3f> worldPoints = LoadXYZCoordinates(activeSet_XYZ_Path);

    // std::cout << " worldPoints: " << std::endl
    //           << worldPoints << std::endl;

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

        std::vector<cv::KeyPoint> keypoints = validation.getKeypoints();
        std::vector<cv::Point2f> imagePoints;

        for (const auto &kp : keypoints)
        {
            imagePoints.push_back(kp.pt);
        }

        // Kameramatrix (intrinsische Parameter)
        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << 100, 0, 50,
                                0, 100, 50,
                                0, 0, 1);

        // Verzerrungskoeffizienten
        cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);

        cv::Mat rvec, tvec;
        EstimatePose(filteredMatches, validation.getKeypoints(), training.getKeypoints(), worldPoints, cameraMatrix, distCoeffs, rvec, tvec);

        cv::imshow("Brute Force Matching", outputImage);

        if (cv::waitKey(1000 / video.getFPS()) == 27) // Esc-Taste beendet die Schleife
            break;
    }

    cv::destroyAllWindows();

    return 0;
}
