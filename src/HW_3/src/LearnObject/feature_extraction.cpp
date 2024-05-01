#include "ComputerVision.h" // Assuming the class definition is in this header

int main()
{
    ComputerVision::FeatureExtraction extractor("/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png");
    
    extractor.createTrackbar();

    extractor.extractFeaturesAndDescriptors();
    
    return 0;
}