#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

namespace ComputerVision
{

class FeatureExtraction {
private:
    cv::Mat inputImage;
    cv::Mat outputImage;
    cv::Mat descriptors;

public:
    FeatureExtraction(const std::string& imagePath) 
    {
        inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (inputImage.empty()) {
            std::cerr << "Error: Image not found or unable to read." << std::endl;
        } else {
            std::cout << "Image loaded successfully." << std::endl;
        } 
    }

    void extractFeaturesAndDescriptors()
    {
        if (inputImage.empty()) 
        {
            std::cerr << "Error: No valid image loaded for feature detection." << std::endl;
            return;
        }

        cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
        
        std::vector<cv::KeyPoint> keypoints;

        detector->detectAndCompute(inputImage, cv::noArray(), keypoints, descriptors);

        cv::drawKeypoints(inputImage, keypoints, outputImage);

        drawIndexesToImage(outputImage, keypoints);

        cv::imwrite("sift_result.jpg", outputImage);

        cv::imshow("Detected Features", outputImage);

        saveToYAML("features.yml", keypoints);
        
        saveToCSV("features.csv", keypoints);
        
        cv::waitKey(0);
    }

    void saveToYAML(const std::string& filename, const std::vector<cv::KeyPoint>& keypoints) 
    {
        cv::FileStorage file(filename, cv::FileStorage::WRITE);
        file << "keypoints" << keypoints;
        file << "descriptors" << descriptors;
        file.release();
    }

    void saveToCSV(const std::string& filename, const std::vector<cv::KeyPoint>& keypoints) 
    {
        std::ofstream csvFile(filename);
        if (csvFile.is_open()) {
            csvFile << "Index,x,y,size,angle,response\n";
            for (int i = 0; i < keypoints.size(); i++) {
                const auto& kp = keypoints[i];
                csvFile << i << "," << kp.pt.x << "," << kp.pt.y << "," << kp.size << "," << kp.angle << "," << kp.response << "\n";
            }
            csvFile.close();
        }
    }

    void drawIndexesToImage(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
    {
        for (size_t i = 0; i < keypoints.size(); ++i) 
        {
            const auto& kp = keypoints[i];
            std::string index = std::to_string(i);
            cv::putText(image,                 // Image
                        index,                       // Text to display
                        kp.pt,                       // Bottom-left corner of the text
                        cv::FONT_HERSHEY_SIMPLEX,    // Font type
                        0.4,                         // Font scale
                        cv::Scalar(0, 255, 0),       // Font color
                        1);                          // Thickness
        }
    }
};
}
