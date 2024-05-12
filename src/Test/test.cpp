#include <opencv2/opencv.hpp>

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

int main()
{
    // Laden des Bildes
    cv::Mat image = loadImage("/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png");

    // Definition des Koordinatensystems
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // Rotationsvektor: Nullvektor (keine Rotation)
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F); // Translationsvektor: Nullvektor (keine Verschiebung)

    // Anpassen des Translationsvektors, um die x- und y-Translation einzustellen
    tvec.at<double>(0) = 0; // x-Komponente
    tvec.at<double>(1) = 0; // y-Komponente
    tvec.at<double>(2) = 0; // z-Komponente

    rvec.at<double>(0) = 0; // x-Rotation
    rvec.at<double>(1) = 0; // y-Rotation
    rvec.at<double>(2) = 0; // z-Rotation

    // Zeichnen Sie das Koordinatensystem
    cv::drawFrameAxes(image, cv::Mat::eye(3, 3, CV_64F), cv::Mat(), rvec, tvec, 100); // 100 ist die Länge der Achsen

    // Anzeigen des Bildes
    cv::imshow("Coordinate System", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
