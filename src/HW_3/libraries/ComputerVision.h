#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>


cv::Mat loadImage(const std::string &imagePath)
{
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    
    if (inputImage.empty())
    {
        std::cerr << "Error: Image not found or unable to read." << std::endl;
    }
    else
    {
        std::cout << "Image loaded successfully." << std::endl;
    }

    return inputImage;
}


namespace ComputerVision
{
    class FeatureExtraction
    {
    private:
        cv::Mat inputImage;
        cv::Mat outputImage;
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints;
        double contrastThreshold; // Variable für den Kontrastschwellenwert
        int contrastThresholdInt; // Variable zum Speichern des Kontrastschwellenwerts als Integer


    public:
        FeatureExtraction(const std::string &imagePath)
        {
            inputImage = loadImage(imagePath);
            contrastThreshold = 0.026;
            contrastThresholdInt = 26;
        }

        // Methode zum Setzen des Kontrastschwellenwerts
        void setContrastThreshold(double value)
        {
            contrastThreshold = value;
            contrastThresholdInt = static_cast<int>(value * 100); // Umrechnen in einen Integer-Wert im Bereich [0, 100]
        }

        // Methode zum Extrahieren von Merkmalen und Deskriptoren
        void extractFeaturesAndDescriptors()
        {
            if (inputImage.empty())
            {
                std::cerr << "Error: No valid image loaded for feature detection." << std::endl;
                return;
            }

            while (true)
            {
                // Merkmale und Deskriptoren mit dem aktuellen Kontrastschwellenwert erkennen und berechnen
                detectAndComputeKeypointsAndDescriptors(inputImage, contrastThreshold, keypoints, descriptors);

                cv::drawKeypoints(inputImage, keypoints, outputImage);

                drawIndexesToImage(outputImage, keypoints);

                cv::imwrite("sift_result.jpg", outputImage);

                cv::imshow("Detected Features", outputImage);

                saveToCSV("activeSet.csv", keypoints);

                // cv::waitKey(0);

                cv::waitKey(10);
            }
            


        }

        // Methode zum Erstellen des Trackbars
        void createTrackbar()
        {
            // Fenster erstellen
            cv::namedWindow("Trackbar");

            // Trackbar erstellen
            cv::createTrackbar("Contrast Threshold", "Trackbar", &contrastThresholdInt, 100, onTrackbar, this);
        }

        cv::Mat getDescriptor()
        {
            return this->descriptors;
        }

        /// @brief getKeypoints
        /// @return  std::vector<cv::KeyPoint> keypoints
        std::vector<cv::KeyPoint> getKeypoints()
        {
            return this->keypoints;
        }

    private:
        // Methode zum Erstellen der Merkmale und Deskriptoren
        void detectAndComputeKeypointsAndDescriptors(const cv::Mat &inputImage, double &contrastThreshold, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
        {
            std::cout << " contrastThreshold: " << contrastThreshold << std::endl;

            // Creating a SIFT feature detector with custom parameters
            cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create(
                0,                 // Number of desired levels in the DoG pyramid construction. 0 means automatic level adjustment.
                3,                 // SIFT control parameter for the number of scales to be investigated in the DoG pyramid construction.
                contrastThreshold, // SIFT control parameter for the threshold to determine corner contrast strength.
                10,                // SIFT control parameter for the number of bins per 1D histogram.
                1.6                // SIFT control parameter for the factor by which the layer sigma is computed.
            );

            detector->detectAndCompute(inputImage, cv::noArray(), keypoints, descriptors);
        }

        // Methode zum Speichern in YAML-Format
        void saveToYAML(const std::string &filename, const std::vector<cv::KeyPoint> &keypoints)
        {
            cv::FileStorage file(filename, cv::FileStorage::WRITE);
            file << "keypoints" << keypoints;
            file << "descriptors" << descriptors;
            file.release();
        }

        // Methode zum Speichern in CSV-Format
        void saveToCSV(const std::string &filename, const std::vector<cv::KeyPoint> &keypoints)
        {
            std::ofstream csvFile(filename);
            if (csvFile.is_open())
            {
                csvFile << "Index,x,y,size,angle,response\n";
                for (int i = 0; i < keypoints.size(); i++)
                {
                    const auto &kp = keypoints[i];
                    csvFile << i << "," << kp.pt.x << "," << kp.pt.y << "," << kp.size << "," << kp.angle << "," << kp.response << "\n";
                }
                csvFile.close();
            }
        }

        // Methode zum Anzeigen der Indizes im Bild
        void drawIndexesToImage(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
        {
            for (size_t i = 0; i < keypoints.size(); ++i)
            {
                const auto &kp = keypoints[i];
                std::string index = std::to_string(i);
                cv::putText(image,                    // Image
                            index,                    // Text to display
                            kp.pt,                    // Bottom-left corner of the text
                            cv::FONT_HERSHEY_SIMPLEX, // Font type
                            0.4,                      // Font scale
                            cv::Scalar(0, 255, 0),    // Font color
                            1);                       // Thickness
            }
        }

        // Statische Callback-Methode für den Trackbar
        static void onTrackbar(int value, void *userdata)
        {
            FeatureExtraction *fe = static_cast<FeatureExtraction *>(userdata);

            double contrastThreshold = static_cast<double>(value) / 100.0;
            
            fe->setContrastThreshold(contrastThreshold);
        }
    };


    class CameraCalibrator
    {
    private:
        std::vector<std::vector<cv::Point3f>> objectPoints;
        std::vector<std::vector<cv::Point2f>> imagePoints;
        cv::Size boardSize;
        cv::Size imageSize;

    public:
        CameraCalibrator(cv::Size boardSize) : boardSize(boardSize) {}

        bool addChessboardPoints(const cv::Mat &image)
        {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corners;
            bool patternFound = cv::findChessboardCorners(gray, boardSize, corners,
                                                          cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

            if (patternFound)
            {
                cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001));

                imagePoints.push_back(corners);

                std::vector<cv::Point3f> objp;
                for (int i = 0; i < boardSize.height; ++i)
                {
                    for (int j = 0; j < boardSize.width; ++j)
                    {
                        objp.push_back(cv::Point3f(j, i, 0));
                    }
                }
                objectPoints.push_back(objp);

                // Anzeigen des Bildes mit erkannten Ecken
                cv::drawChessboardCorners(image, boardSize, corners, patternFound);
                cv::imshow("Detected Chessboard", image);
                cv::waitKey(0);

                return true;
            }

            return false;
        }

        void calibrate(const cv::Mat &image)
        {
            imageSize = image.size();
            cv::Mat cameraMatrix, distCoeffs, R, T;

            cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, R, T);

            std::cout << "Camera Matrix:\n"
                      << cameraMatrix << std::endl;
            std::cout << "Distortion Coefficients:\n"
                      << distCoeffs << std::endl;
            std::cout << "Rotation Vector:\n"
                      << R << std::endl;
            std::cout << "Translation Vector:\n"
                      << T << std::endl;
        }
    };

}
