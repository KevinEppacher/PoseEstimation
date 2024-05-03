#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "ComputerVision.h"

int main()
{
    // Öffne das Video
    cv::VideoCapture cap("/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4");

    // Überprüfe, ob das Video erfolgreich geöffnet wurde
    if (!cap.isOpened())
    {
        std::cerr << "Fehler beim Öffnen des Videos!" << std::endl;
        return -1;
    }

    cv::Mat frame, grayFrame;

    // std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";

    // std::string validateImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png";

    // ComputerVision::FeatureExtraction validation(validateImagePath);

    // ComputerVision::FeatureExtraction training(trainingImagePath);

    // validation.computeKeypointsAndDescriptors();

    // training.computeKeypointsAndDescriptors();

    // ComputerVision::computeBruteForceMatching(training, validation);

    // Schleife zum Durchlaufen jedes Frames im Video
    while (true)
    {
        // // Lese das nächste Frame
        cv::Mat frame;
        cap >> frame;

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";

        ComputerVision::FeatureExtraction training(trainingImagePath);

        ComputerVision::FeatureExtraction validation(grayFrame);

        bool trackbar = false;

        training.computeKeypointsAndDescriptors(trackbar);

        validation.computeKeypointsAndDescriptors(trackbar);

        cv::Mat outputImage;

        ComputerVision::computeBruteForceMatching(training, validation, outputImage);

        cv::imshow("Brute Force Matching", outputImage);

        // cv::waitKey(100);

        if (cv::waitKey(10) == 27)
            break;
    }

    // Schließe das Video
    cap.release();
    cv::destroyAllWindows();

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