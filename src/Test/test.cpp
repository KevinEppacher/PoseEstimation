#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

int main() {
    // 3D-Punkte im Weltkoordinatensystem
    std::vector<cv::Point3f> objectPoints = {cv::Point3f(0, 0, 0),
                                              cv::Point3f(1, 0, 0),
                                              cv::Point3f(0, 1, 0),
                                              cv::Point3f(1, 1, 0)};

    // Ecken des Musters im Bild
    std::vector<cv::Point2f> imagePoints = {cv::Point2f(10, 10),
                                             cv::Point2f(20, 10),
                                             cv::Point2f(10, 20),
                                             cv::Point2f(20, 20),
                                             cv::Point2f(20, 20)};

    // Kameramatrix (intrinsische Parameter)
    cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << 100, 0, 50,
                                                     0, 100, 50,
                                                     0, 0, 1);

    // Verzerrungskoeffizienten
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);

    // solvePnP aufrufen, um die Pose zu schätzen
    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    // Drucken der Ergebnisse
    std::cout << "Rotation vector (rvec):" << std::endl;
    std::cout << rvec << std::endl;
    std::cout << "Translation vector (tvec):" << std::endl;
    std::cout << tvec << std::endl;

    return 0;
}
