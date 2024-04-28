#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Kameraobjekt erstellen. 0 ist normalerweise die ID der ersten angeschlossenen Kamera
    cv::VideoCapture cap(0);

    // Überprüfen, ob die Kamera korrekt initialisiert wurde
    if (!cap.isOpened()) {
        std::cerr << "Fehler: Kamera konnte nicht geöffnet werden." << std::endl;
        return -1;
    }

    // Fenster erstellen, in dem das Video angezeigt wird
    cv::namedWindow("Kamera", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat frame;
        // Einen neuen Frame von der Kamera holen
        cap >> frame;

        // Überprüfen, ob der Frame leer ist (z.B. wenn die Kamera getrennt wurde)
        if (frame.empty()) {
            std::cerr << "Fehler: Leerer Frame empfangen." << std::endl;
            break;
        }

        // Den Frame im Fenster anzeigen
        cv::imshow("Kamera", frame);

        // Warten auf 'q' Taste, um die Schleife zu beenden
        if (cv::waitKey(25) == 'q') {
            break;
        }
    }

    // Kamera freigeben
    cap.release();
    // Alle geöffneten HighGUI Fenster schließen
    cv::destroyAllWindows();

    return 0;
}
