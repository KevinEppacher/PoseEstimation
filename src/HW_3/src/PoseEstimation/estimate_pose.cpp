#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include "ComputerVision.h"

int main()
{

    std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";

    std::string validateImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png";

    ComputerVision::FeatureExtraction validation(validateImagePath);

    ComputerVision::FeatureExtraction training(trainingImagePath);

    validation.computeKeypointsAndDescriptors();

    training.computeKeypointsAndDescriptors();

    ComputerVision::computeBruteForceMatching(training, validation);


    return 0;
}

// int main()
// {

//     // Laden der Bilder
//     cv::Mat img1 = cv::imread("/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png", cv::IMREAD_GRAYSCALE);
//     cv::Mat img2 = cv::imread("/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png", cv::IMREAD_GRAYSCALE);

//     // Erzeugen eines SIFT-Detektors
//     cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

//     // Finden von Key-Points und Deskriptoren für beide Bilder
//     std::vector<cv::KeyPoint> keypoints1, keypoints2;
//     cv::Mat descriptors1, descriptors2;
//     sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
//     sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

//     // Erzeugen eines Brute-Force-Matchers
//     cv::BFMatcher bf(cv::NORM_L2);

//     // Durchführen von Brute-Force-Matching
//     std::vector<cv::DMatch> matches;
//     bf.match(descriptors1, descriptors2, matches);

//     // Sortieren der Matches nach Distanz
//     std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

//     // Zeigen der ersten 10 Matches
//     cv::Mat img_matches;
//     cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//     cv::imshow("Matches", img_matches);
//     cv::waitKey(0);
//     cv::destroyAllWindows();

//     return 0;
// }