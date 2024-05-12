#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

/**
 * @brief The `cv::Mat` class represents a multi-dimensional dense numerical array used in OpenCV for image processing.
 *
 * It is used to store and manipulate images and other multi-dimensional data. The `cv::Mat` class provides various
 * methods and operators for accessing and manipulating the pixel values of an image.
 */
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

/**
 * @brief A vector of 3D points in the OpenCV library.
 *
 * This vector is used to store a collection of 3D points, where each point is represented by
 * a cv::Point3f object. It provides methods for adding, accessing, and manipulating the points
 * in the vector.
 */
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

    /**
     * @class FeatureExtraction
     * @brief A class that performs feature extraction on an input image using the SIFT algorithm.
     */
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
        /**
         * @brief Constructs a FeatureExtraction object with the given image path.
         *
         * @param imagePath The path to the input image.
         */
        FeatureExtraction(const std::string &imagePath)
        {
            inputImage = loadImage(imagePath);
            contrastThreshold = 0.26;
            contrastThresholdInt = 26;
        }

        FeatureExtraction(){};

        /**
         * @brief Extracts features from an image.
         *
         * This function takes an input image and extracts features using a specific algorithm.
         * The extracted features are returned as a vector of feature descriptors.
         *
         * @param image The input image from which features are to be extracted.
         * @param algorithm The algorithm used for feature extraction.
         * @return A vector of feature descriptors.
         */
        FeatureExtraction(cv::Mat &inputImage) : inputImage(inputImage){};

        /**
         * @brief Sets the contrast threshold for feature extraction.
         *
         * This function sets the contrast threshold for feature extraction.
         * The contrast threshold determines the minimum contrast required for a pixel to be considered a feature.
         *
         * @param value The contrast threshold value.
         */
        void setContrastThreshold(double value)
        {
            contrastThreshold = value;
            contrastThresholdInt = static_cast<int>(value * 100); // Convert to an integer value in the range [0, 100]
        }

        /**
         * @brief Computes keypoints and descriptors.
         *
         * This function computes keypoints and descriptors using the SIFT algorithm.
         * If withTrackbar is true, it displays a trackbar to adjust the contrast threshold interactively.
         * If withTrackbar is false, it computes the keypoints and descriptors without displaying the trackbar.
         *
         * @param withTrackbar Flag indicating whether to display the trackbar or not.
         */
        void computeKeypointsAndDescriptors(bool withTrackbar)
        {
            bool isRunning = true;

            if (withTrackbar)
            {
                // Create window
                cv::namedWindow("Trackbar");

                // Create trackbar
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
                        isRunning = false; // Exit the loop
                    }
                }

                cv::imwrite("sift_result.jpg", outputImage);

                saveKeypointsToCSV("activeSet.csv", keypoints);

                // cv::destroyAllWindows();
            }
            else
            {
                detectAndComputeKeypointsAndDescriptors(inputImage, contrastThreshold, keypoints, descriptors);

                cv::drawKeypoints(inputImage, keypoints, outputImage);

                drawIndexesToImage(outputImage, keypoints);
            }
        }

        /**
         * @brief Filters keypoints and descriptors based on given indices.
         *
         * This function filters the keypoints and descriptors based on the given indices.
         * Only the keypoints and descriptors with indices in the provided vector will be kept.
         *
         * @param indices The indices of the keypoints and descriptors to keep.
         */
        void filterKeypointsAndDescriptor(const std::vector<int> &indices)
        {
            std::vector<cv::KeyPoint> newKeypoints;
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

        /**
         * @brief Filters keypoints and descriptors based on indices provided in a CSV file.
         *
         * This function filters the keypoints and descriptors based on the indices provided in a CSV file.
         * Only the keypoints and descriptors with indices in the CSV file will be kept.
         *
         * @param csvPath The path to the CSV file containing the indices.
         * @return A vector of filtered indices.
         * @throws std::runtime_error if the CSV file cannot be opened.
         */
        std::vector<int> filterKeypointsAndDescriptor(const std::string &csvPath)
        {
            std::ifstream file(csvPath);

            if (!file.is_open())
                throw std::runtime_error("Could not open CSV file");

            std::vector<int> indices;
            std::string line, value;

            std::getline(file, line);

            while (std::getline(file, line))
            {
                std::stringstream ss(line);

                std::getline(ss, value, ',');

                // Überprüfen Sie, ob der String `value` eine Ganzzahl darstellt
                if (!value.empty() && std::all_of(value.begin(), value.end(), ::isdigit))
                {
                    indices.push_back(std::stoi(value));
                }
                else
                {
                    std::cerr << "Warning: Invalid value in CSV file: " << value << std::endl;
                }
            }

            file.close();

            return indices;
        }

        /**
         * Retrieves the input image.
         *
         * @return The input image as a cv::Mat object.
         */
        cv::Mat getImage()
        {
            return inputImage;
        }

        /**
         * @brief Get the descriptor.
         *
         * @return cv::Mat The descriptor matrix.
         */
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
        /**
         * @brief Detects keypoints and computes descriptors.
         *
         * This function detects keypoints and computes descriptors using the SIFT algorithm.
         *
         * @param inputImage The input image from which keypoints and descriptors are to be computed.
         * @param contrastThreshold The contrast threshold for determining corner contrast strength.
         * @param keypoints The output vector of keypoints.
         * @param descriptors The output matrix of descriptors.
         */
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

        /**
         * @brief Saves keypoints and descriptors to a YAML file.
         *
         * This function saves the keypoints and descriptors to a YAML file.
         *
         * @param filename The name of the YAML file to save.
         * @param keypoints The vector of keypoints to save.
         */
        void saveToYAML(const std::string &filename, const std::vector<cv::KeyPoint> &keypoints)
        {
            cv::FileStorage file(filename, cv::FileStorage::WRITE);
            file << "keypoints" << keypoints;
            file << "descriptors" << descriptors;
            file.release();
        }

        /**
         * @brief Saves keypoints to a CSV file.
         *
         * This function saves the keypoints to a CSV file.
         *
         * @param filename The name of the CSV file to save.
         * @param keypoints The vector of keypoints to save.
         */
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

        /**
         * @brief Draws indexes on the image for each keypoint.
         *
         * This function draws indexes on the image for each keypoint.
         *
         * @param image The image on which to draw the indexes.
         * @param keypoints The vector of keypoints.
         */
        void drawIndexesToImage(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
        {
            for (size_t i = 0; i < keypoints.size(); ++i)
            {
                cv::putText(image, std::to_string(i), keypoints[i].pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }

        /**
         * @brief Static callback method for the trackbar.
         *
         * This function is a static callback method for the trackbar.
         *
         * @param value The value of the trackbar.
         * @param userdata The user data passed to the callback.
         */
        static void onTrackbar(int value, void *userdata)
        {
            FeatureExtraction *fe = static_cast<FeatureExtraction *>(userdata);
            double contrastThreshold = static_cast<double>(value) / 100.0;
            fe->setContrastThreshold(contrastThreshold);
        }
    };

    class Video
    {
    private:
        cv::VideoCapture cap;
        std::vector<cv::Mat> frames;
        cv::Mat frame;
        double fps;

    public:
        /**
         * @brief Constructs a Video object from a video file.
         *
         * This constructor opens the video file specified by the given path and initializes the Video object.
         * It reads all the frames from the video and stores them in the frames vector.
         * It also retrieves the frames per second (FPS) of the video.
         *
         * @param videoPath The path to the video file.
         * @throws std::runtime_error if the video file cannot be opened or if no frames are found in the video.
         */
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

        /**
         * @brief Puts the frames per second (FPS) on the image.
         *
         * This function puts the frames per second (FPS) on the image.
         *
         * @param image The image on which to put the FPS.
         */
        void putFpsOnImage(cv::Mat &image)
        {
            std::ostringstream oss;
            oss << "FPS: " << fps;
            cv::putText(image, oss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        /**
         * @brief Retrieves the frames per second (FPS) of the video.
         *
         * @return The frames per second (FPS) as a double value.
         */
        double getFPS()
        {
            return fps;
        }

        /**
         * @brief Retrieves the frames of the video.
         *
         * @return The frames of the video as a vector of cv::Mat objects.
         */
        std::vector<cv::Mat> getFrames()
        {
            return frames;
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
        std::string videoPath;

    public:
        /**
         * @brief Constructs a CameraCalibrator object with the specified board size.
         *
         * This constructor initializes a CameraCalibrator object with the specified board size.
         *
         * @param boardSize The size of the chessboard used for calibration.
         */
        CameraCalibrator(cv::Size boardSize) : boardSize(boardSize) {}

        /**
         * @brief Constructs a CameraCalibrator object with the specified video path.
         *
         * This constructor initializes a CameraCalibrator object with the specified video path.
         *
         * @param videoPath The path to the video file used for calibration.
         */
        CameraCalibrator(std::string videoPath) : videoPath(videoPath) {}

        /**
         * @brief Default constructor for CameraCalibrator.
         *
         * This constructor creates a CameraCalibrator object with default parameters.
         */
        CameraCalibrator() {}

        /**
         * @brief Sets the video path for the CameraCalibrator object.
         *
         * This function sets the video path for the CameraCalibrator object.
         *
         * @param videoPath The path to the video file.
         */
        void setVideoPath(const std::string &videoPath)
        {
            this->videoPath = videoPath;
        }

        /**
         * @brief Loads a video and adds chessboard points for calibration.
         *
         * This function loads a video from the specified path and adds chessboard points for calibration.
         * It retrieves the frames from the video and calls the addChessboardPoints function for each frame.
         *
         * @param videoPath The path to the video file.
         * @param showCalibrationImages Flag indicating whether to show the calibration images.
         */
        void loadVideoAndAddChessboardPoints(const std::string &videoPath, bool showCalibrationImages)
        {
            Video video(videoPath);
            std::vector<cv::Mat> frames = video.getFrames();

            for (auto &frame : frames)
            {
                if (!frame.empty())
                {
                    addChessboardPoints(frame, showCalibrationImages);
                }
            }
        }

        /**
         * @brief Loads images from a directory and adds chessboard points for calibration.
         *
         * This function loads images from the specified directory and adds chessboard points for calibration.
         * It retrieves the images from the directory and calls the addChessboardPoints function for each image.
         *
         * @param imageDirectory The directory containing the images.
         * @param showCalibrationImages Flag indicating whether to show the calibration images.
         */
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

        /**
         * @brief Adds chessboard points for calibration.
         *
         * This function adds chessboard points for calibration to the objectPoints and imagePoints vectors.
         * It detects the chessboard corners in the input image and adds the corresponding 3D and 2D points to the vectors.
         *
         * @param image The input image.
         * @param showCalibrationImages Flag indicating whether to show the calibration images.
         * @return True if the chessboard is detected in the image, false otherwise.
         */
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

                std::cout << "Chessboard detected in image: " << imagePoints.size() << std::endl;

                if (showCalibrationImages == true)
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

        /**
         * @brief Calibrates the camera.
         *
         * This function calibrates the camera using the objectPoints and imagePoints vectors.
         * It calculates the camera matrix and distortion coefficients using the calibrateCamera function from OpenCV.
         * The resulting camera matrix and distortion coefficients are stored in the cameraMatrix and distCoeffs variables.
         */
        void calibrate()
        {
            // Calculate the size of the image
            cv::Size imageSize(imagePoints[0].size(), imagePoints.size());

            // Declare variables for rotation and translation vectors
            cv::Mat R, T;

            // Calibrate the camera
            cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, R, T);

            // Print the camera matrix and distortion coefficients
            std::cout << "Camera Matrix:\n"
                      << cameraMatrix << std::endl;
            std::cout << "Distortion Coefficients:\n"
                      << distCoeffs << std::endl;
        }

        /**
         * @brief Retrieves the camera matrix.
         *
         * This function retrieves the camera matrix used for camera calibration.
         *
         * @return The camera matrix as a cv::Mat object.
         */
        cv::Mat getCameraMatrix() const
        {
            return cameraMatrix;
        }

        /**
         * @brief Retrieves the distortion coefficients.
         *
         * This function retrieves the distortion coefficients used for camera calibration.
         *
         * @return The distortion coefficients as a cv::Mat object.
         */
        cv::Mat getDistCoeffs() const
        {
            return distCoeffs;
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
        /**
         * @brief Constructs a Matcher object with the specified training and validation feature extractions.
         *
         * This constructor initializes a Matcher object with the specified training and validation feature extractions.
         * It retrieves the keypoints, descriptors, and images from the training and validation feature extractions.
         *
         * @param training The training FeatureExtraction object.
         * @param validation The validation FeatureExtraction object.
         */
        Matcher(FeatureExtraction &training, FeatureExtraction &validation)
        {
            // Retrieve keypoints, descriptors, and images from training and validation feature extractions
            trainingKeypoints = training.getKeypoints();
            validationKeypoints = validation.getKeypoints();
            trainingDescription = training.getDescriptor();
            validationDescription = validation.getDescriptor();
            trainingImage = training.getImage();
            validationImage = validation.getImage();
        }

        ~Matcher() {}

        /**
         * @brief Matches the features between the training and validation feature extractions.
         *
         * This function matches the features between the training and validation feature extractions.
         * It calls the computeMatches function to compute the matches and assigns the result to the matches vector.
         *
         * @return The vector of matches as a vector of cv::DMatch objects.
         */
        std::vector<cv::DMatch> matchFeatures()
        {
            return matches = computeMatches(trainingDescription, validationDescription);
        }

        /**
         * @brief Computes the matches between the training and validation feature descriptors.
         *
         * This function computes the matches between the training and validation feature descriptors using the BFMatcher.
         * It sorts the matches based on their distance and returns the sorted matches.
         *
         * @param trainingDescription The descriptor of the training feature extraction.
         * @param validationDescription The descriptor of the validation feature extraction.
         * @return The vector of matches as a vector of cv::DMatch objects.
         */
        std::vector<cv::DMatch> computeMatches(const cv::Mat &trainingDescription, const cv::Mat &validationDescription)
        {
            cv::BFMatcher bf(cv::NORM_L2);

            std::vector<cv::DMatch> matches;
            bf.match(trainingDescription, validationDescription, matches);

            std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b)
                      { return a.distance < b.distance; });

            return matches;
        }

        /**
         * @brief Filters the matches based on a distance threshold.
         *
         * This function filters the matches based on a distance threshold.
         * It iterates through each match and checks if its distance is less than the specified threshold.
         * If the distance is less than the threshold, the match is added to the vector of good matches.
         *
         * @param TH The distance threshold.
         * @return The vector of filtered matches as a vector of cv::DMatch objects.
         */
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

        /**
         * @brief Draws the matches between the training and validation feature extractions.
         *
         * This function draws the matches between the training and validation feature extractions.
         * It takes the vector of matches, the training image, the validation image, and the output image as parameters.
         * It uses the cv::drawMatches function to draw the matches on the output image.
         *
         * @param matches The vector of matches as a vector of cv::DMatch objects.
         * @param trainingImage The training image.
         * @param validationImage The validation image.
         * @param imgMatches The output image where the matches will be drawn.
         */
        void drawMatches(const std::vector<cv::DMatch> &matches, cv::Mat &imgMatches)
        {
            cv::drawMatches(trainingImage, trainingKeypoints, validationImage, validationKeypoints, matches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }
    };

    class PoseEstimator
    {
    public:
        /**
         * @brief Constructs a PoseEstimator object with the specified camera matrix and distortion coefficients.
         *
         * This constructor initializes a PoseEstimator object with the specified camera matrix and distortion coefficients.
         *
         * @param cameraMatrix The camera matrix as a cv::Mat object.
         * @param distCoeffs The distortion coefficients as a cv::Mat object.
         */
        PoseEstimator(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) : cameraMatrix(cameraMatrix), distCoeffs(distCoeffs) {}

        /**
         * @brief Constructs a PoseEstimator object with the camera calibration.
         *
         * This constructor initializes a PoseEstimator object with the camera calibration.
         * It retrieves the camera matrix and distortion coefficients from the camera calibration object.
         *
         * @param cameraCalibration The CameraCalibrator object used for camera calibration.
         */
        PoseEstimator(ComputerVision::CameraCalibrator cameraCalibration) : cameraMatrix(cameraCalibration.getCameraMatrix()), distCoeffs(cameraCalibration.getDistCoeffs()) {}

        /**
         * @brief Estimates the pose based on the matches, keypoints, and object points.
         *
         * This function estimates the pose based on the matches, keypoints, and object points.
         * It calculates the image points and selected object points from the matches.
         * If there are not enough points, it returns false.
         * It then calls the solvePnP function to estimate the pose using the selected object points and image points.
         * If the pose estimation fails, it returns false.
         * Otherwise, it prints the translation and rotation vectors and returns true.
         *
         * @param matches The vector of matches as a vector of cv::DMatch objects.
         * @param keypoints The keypoints from the validation feature extraction.
         * @param objectPoints The object points used for camera calibration.
         * @return True if the pose estimation is successful, false otherwise.
         */
        bool estimatePose(const std::vector<cv::DMatch> &matches,
                          const std::vector<cv::KeyPoint> &keypoints,
                          const std::vector<cv::Point3f> &objectPoints)
        {
            std::vector<cv::Point2f> imagePoints;
            std::vector<cv::Point3f> selectedObjectPoints;

            for (const auto &match : matches)
            {
                imagePoints.push_back(keypoints[match.trainIdx].pt);
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

        std::vector<cv::Point2f> extractImagePointsFromKeypoints(const std::vector<cv::KeyPoint> &keypoints)
        {
            std::vector<cv::Point2f> imagePoints;

            for (const auto &kp : keypoints)
            {
                imagePoints.push_back(kp.pt);
            }

            return imagePoints;
        }

    private:
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat rvec, tvec;
        ComputerVision::CameraCalibrator camera;
    };

}
