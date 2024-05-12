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
    cv::Mat inputImage = cv::imread(imagePath);

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

std::vector<cv::Point3f> LoadXYZCoordinates(const std::string &filePath)
{
    std::vector<cv::Point3f> coordinates;

    // Datei öffnen
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Fehler beim Öffnen der Datei: " << filePath << std::endl;
        return coordinates;
    }

    // Header-Zeile überspringen
    std::string line;
    std::getline(file, line);

    // Zeilen einlesen und verarbeiten
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        int index;
        char comma;
        float x, y, z;

        // CSV-Werte einlesen
        if (!(iss >> index >> comma >> x >> comma >> y >> comma >> z))
        {
            std::cerr << "Ungültiges CSV-Format in Zeile: " << line << std::endl;
            continue;
        }

        // Punkt erstellen und in das Vektor einfügen
        coordinates.push_back(cv::Point3f(x, y, z));
    }

    // Datei schließen
    file.close();

    return coordinates;
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
        double contrastThreshold;
        int contrastThresholdInt;

    public:
        FeatureExtraction(const std::string &imagePath)
        {
            inputImage = loadImage(imagePath);
            contrastThreshold = 0.26;
            contrastThresholdInt = 26;
        }

        FeatureExtraction(){};

        FeatureExtraction(const cv::Mat &inputImage) : inputImage(inputImage){};

        void setContrastThreshold(double value)
        {
            contrastThreshold = value;
            contrastThresholdInt = static_cast<int>(value * 100); // Umrechnen in einen Integer-Wert im Bereich [0, 100]
        }

        void computeKeypointsAndDescriptors(bool withTrackbar)
        {

            bool isRunning = true;

            if (withTrackbar)
            {
                // Fenster erstellen
                cv::namedWindow("Trackbar");

                // Trackbar erstellen
                cv::createTrackbar("Contrast Threshold", "Trackbar", &contrastThresholdInt, 100, onTrackbar, this);

                while (isRunning)
                {
                    detectAndComputeKeypointsAndDescriptors(inputImage, contrastThreshold, keypoints, descriptors);

                    cv::drawKeypoints(inputImage, keypoints, outputImage);

                    drawIndexesToImage(outputImage, keypoints);

                    cv::imshow("Calibrate Hyperparameter Contstrast Threshold", outputImage);

                    int key = cv::waitKey(10);

                    if (key != -1)
                    {
                        std::cout << "Key pressed: " << key << std::endl;
                        isRunning = false; // Beende die Schleife
                    }
                }

                cv::imwrite("sift_result.jpg", outputImage);

                saveKeypointsToCSV("activeSet.csv", keypoints);

                cv::destroyAllWindows();
            }
            else
            {
                detectAndComputeKeypointsAndDescriptors(inputImage, contrastThreshold, keypoints, descriptors);

                cv::drawKeypoints(inputImage, keypoints, outputImage);

                drawIndexesToImage(outputImage, keypoints);
            }
        }

        void filterKeypointsAndDescriptor(const std::vector<int> &indices)
        {
            std::vector<cv::KeyPoint> newKeypoints;
            ;
            cv::Mat newDescriptors;

            for (int index : indices)
            {
                if (index >= 0 && index < keypoints.size())
                {
                    newKeypoints.push_back(keypoints[index]);
                    newDescriptors.push_back(descriptors.row(index));
                }
                else
                {
                    std::cerr << "Warning: index " << index << " is out of range." << std::endl;
                }
            }
            keypoints = newKeypoints;
            descriptors = newDescriptors;
        }

        cv::Mat getImage()
        {
            return inputImage;
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
        void saveKeypointsToCSV(const std::string &filename, const std::vector<cv::KeyPoint> &keypoints)
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
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;

    public:
        CameraCalibrator(cv::Size boardSize) : boardSize(boardSize) {}
        CameraCalibrator() {}

        void loadImagesAndAddChessboardPoints(const std::string &imageDirectory, bool showCalibrationImages)
        {
            std::vector<cv::String> images;
            std::string path = imageDirectory + "/*.jpeg";
            showCalibrationImages = showCalibrationImages;
            cv::glob(path, images);

            for (const auto &imagePath : images)
            {
                cv::Mat image = cv::imread(imagePath);

                if (!image.empty())
                {
                    addChessboardPoints(image, showCalibrationImages);
                }
            }
        }

        bool addChessboardPoints(const cv::Mat &image, bool showCalibrationImages)
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

                if(showCalibrationImages == true)
                {
                    // Draw detected chessboard corners
                    cv::drawChessboardCorners(image, boardSize, corners, patternFound);
                    cv::imshow("Detected Chessboard", image);
                    cv::waitKey(0);
                }


                return true;
            }

            return false;
        }

        void calibrate()
        {
            cv::Size imageSize(imagePoints[0].size(), imagePoints.size());
            cv::Mat R, T;

            cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, R, T);

            std::cout << "Camera Matrix:\n"
                      << cameraMatrix << std::endl;
            std::cout << "Distortion Coefficients:\n"
                      << distCoeffs << std::endl;
        }

        cv::Mat getCameraMatrix() const
        {
            return cameraMatrix;
        }

        cv::Mat getDistCoeffs() const
        {
            return distCoeffs;
        }
    };

    class Video
    {
    private:
        cv::VideoCapture cap;
        double fps;
        std::vector<cv::Mat> frames;
        cv::Mat frame;

    public:
        Video(const std::string &videoPath) : cap(videoPath)
        {
            cap.open(videoPath);
            if (!cap.isOpened())
            {
                throw std::runtime_error("Fehler beim Öffnen des Videos!");
            }

            fps = cap.get(cv::CAP_PROP_FPS);

            while (cap.read(frame))
            {
                frames.push_back(frame.clone());
            }

            if (frames.empty())
            {
                throw std::runtime_error("Keine Frames im Video gefunden!");
            }
        }

        ~Video() {}

        void putFpsOnImage(cv::Mat &image)
        {
            std::ostringstream oss;
            oss << "FPS: " << fps;
            cv::putText(image, oss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        double getFPS()
        {
            return fps;
        }

        std::vector<cv::Mat> getFrames()
        {
            return frames;
        }
    };

    class Matcher
    {
    private:
        std::vector<cv::KeyPoint> trainingKeypoints, validationKeypoints;
        cv::Mat trainingDescription, validationDescription;
        cv::Mat trainingImage, validationImage;
        cv::Mat imgMatches;
        std::vector<cv::DMatch> matches;

    public:
        Matcher(FeatureExtraction &training, FeatureExtraction &validation)
        {
            trainingKeypoints = training.getKeypoints();
            validationKeypoints = validation.getKeypoints();
            trainingDescription = training.getDescriptor();
            validationDescription = validation.getDescriptor();
            trainingImage = training.getImage();
            validationImage = validation.getImage();
        }

        ~Matcher() {}

        std::vector<cv::DMatch> matchFeatures()
        {
            return matches = computeMatches(trainingDescription, validationDescription);
        }

        std::vector<cv::DMatch> computeMatches(const cv::Mat &trainingDescription, const cv::Mat &validationDescription)
        {
            cv::BFMatcher bf(cv::NORM_L2);

            bf.match(trainingDescription, validationDescription, matches);

            std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b)
                      { return a.distance < b.distance; });

            return matches;
        }

        std::vector<cv::DMatch> filterMatches(const double &TH)
        {
            std::vector<cv::DMatch> good_matches;

            for (const auto &match : matches)
            {
                if (match.distance < TH)
                {
                    good_matches.push_back(match);
                }
            }

            return good_matches;
        }

        void drawMatches(const std::vector<cv::DMatch> &matches, cv::Mat &imgMatches)
        {
            cv::drawMatches(trainingImage, trainingKeypoints, validationImage, validationKeypoints, matches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }
    };

    class PoseEstimator
    {
    public:
        PoseEstimator(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) : cameraMatrix(cameraMatrix), distCoeffs(distCoeffs) {}
        PoseEstimator(ComputerVision::CameraCalibrator cameraCalibration): cameraMatrix(cameraCalibration.getCameraMatrix()), distCoeffs(cameraCalibration.getDistCoeffs()){}


    bool estimatePose(const std::vector<cv::DMatch> &matches,
                    const std::vector<cv::KeyPoint> &keypoints1,
                    const std::vector<cv::KeyPoint> &keypoints2,
                    const std::vector<cv::Point3f> &objectPoints)
    {
        std::vector<cv::Point2f> imagePoints;
        std::vector<cv::Point3f> selectedObjectPoints;

        for (const auto &match : matches)
        {
            imagePoints.push_back(keypoints2[match.trainIdx].pt);
            selectedObjectPoints.push_back(objectPoints[match.queryIdx]);
        }

        if (imagePoints.size() < 4 || selectedObjectPoints.size() < 4)
        {
            std::cerr << "Not enough points to estimate pose." << std::endl;
            return false;
        }

        cv::Mat local_rvec, local_tvec;

        bool success = cv::solvePnP(selectedObjectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

        if (!success)
        {
            std::cerr << "Pose estimation failed." << std::endl;
            return false;
        }

        std::cout << "Translation Vector:\n"
                << tvec << std::endl;
        std::cout << "Rotation Vector:\n"
                << rvec << std::endl;

        

        return true;
    }

    void drawCoordinateSystem(cv::Mat &image)
    {
        // Wenn die Rotation und Translation nicht berechnet werden konnten, wird die Methode abgebrochen
        if (rvec.empty() || tvec.empty())
        {
            return;
        }

        tvec.at<double>(0) *= 1000; // x-Komponente
        tvec.at<double>(1) *= 1000; // x-Komponente
        tvec.at<double>(2) *= 1000; // x-Komponente

        std::cout << "Translation Vector Coordinate:\n"
                << tvec << std::endl;


        // Zeichnen Sie das Koordinatensystem
        cv::drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 100); // 100 ist die Länge der Achsen

        // cv::waitKey(0);

    }

    cv::Mat getRvec()
    {
        return rvec;
    }

    cv::Mat getTvec()
    {
        return tvec;
    }


    private:
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat rvec, tvec;
        ComputerVision::CameraCalibrator camera;
    };


}
