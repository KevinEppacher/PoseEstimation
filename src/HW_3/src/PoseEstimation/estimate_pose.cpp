#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ComputerVision.h" // Annahme: ComputerVision.h enthält die Deklarationen der Klassen und Funktionen für Feature-Extraktion und Brute-Force-Matching

int main() {
    // Pfad zum Video
    std::string videoPath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4";

    // Video-Capture-Objekt erstellen
    cv::VideoCapture cap(videoPath);

    // Überprüfen, ob das Video geöffnet werden kann
    if (!cap.isOpened()) {
        std::cerr << "Fehler beim Öffnen des Videos!" << std::endl;
        return -1;
    }

    // Vektor zur Speicherung der Frames
    std::vector<cv::Mat> frames;

    cv::Mat frame, grayFrame;

    // Alle Frames aus dem Video lesen und im Vektor speichern
    while (cap.read(frame)) {
        frames.push_back(frame.clone()); // Kopie des Frames hinzufügen, um unerwartetes Verhalten zu vermeiden
    }

    // Überprüfen, ob Frames vorhanden sind
    if (frames.empty()) {
        std::cerr << "Keine Frames im Video gefunden!" << std::endl;
        return -1;
    }

    // Iteration über alle Frames im Vektor
    for (auto frame : frames) {
        // Frame in Graustufen umwandeln
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        std::string trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png";

        // Feature-Extraktion für das Trainingsbild
        ComputerVision::FeatureExtraction training(trainingImagePath);
        training.computeKeypointsAndDescriptors(false);

        // Feature-Extraktion für das aktuelle Frame
        ComputerVision::FeatureExtraction validation(grayFrame);
        validation.computeKeypointsAndDescriptors(false);

        // Ausführen des Brute-Force-Matchings und Anzeigen des Ergebnisses
        cv::Mat outputImage;
        ComputerVision::computeBruteForceMatching(training, validation, outputImage);
        cv::imshow("Brute Force Matching", outputImage);

        // Auf Benutzereingabe warten
        if (cv::waitKey(10) == 27) // Esc-Taste beendet die Schleife
            break;
    }

    // Fenster schließen
    cv::destroyAllWindows();

    return 0;
}
