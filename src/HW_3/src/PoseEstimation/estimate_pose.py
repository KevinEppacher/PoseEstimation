import cv2
import numpy as np

class FeatureExtraction:
    def __init__(self, imagePath=None):
        self.inputImage = None
        self.outputImage = None
        self.descriptors = None
        self.keypoints = []
        self.contrastThreshold = 0.26
        self.contrastThresholdInt = 26

        if imagePath:
            self.inputImage = self.loadImage(imagePath)

    def loadImage(self, imagePath):
        return cv2.imread(imagePath)

    def setContrastThreshold(self, value):
        self.contrastThreshold = value
        self.contrastThresholdInt = int(value * 100)

    def computeKeypointsAndDescriptors(self, withTrackbar=False):
        isRunning = True

        if withTrackbar:
            cv2.namedWindow("Trackbar")
            cv2.createTrackbar("Contrast Threshold", "Trackbar", self.contrastThresholdInt, 100, self.onTrackbar)

            while isRunning:
                self.detectAndComputeKeypointsAndDescriptors(self.inputImage, self.contrastThreshold, self.keypoints, self.descriptors)

                self.outputImage = self.inputImage.copy()
                self.drawIndexesToImage(self.outputImage, self.keypoints)

                cv2.imshow("Calibrate Hyperparameter Contstrast Threshold", self.outputImage)

                key = cv2.waitKey(10)
                if key != -1:
                    print("Key pressed:", key)
                    isRunning = False

            cv2.imwrite("sift_result.jpg", self.outputImage)
            self.saveToCSV("activeSet.csv", self.keypoints)
            cv2.destroyAllWindows()
        else:
            self.detectAndComputeKeypointsAndDescriptors(self.inputImage, self.contrastThreshold, self.keypoints, self.descriptors)
            self.outputImage = self.inputImage.copy()
            self.drawIndexesToImage(self.outputImage, self.keypoints)

    def detectAndComputeKeypointsAndDescriptors(self, inputImage, contrastThreshold, keypoints, descriptors):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(inputImage, None)

        self.keypoints = keypoints
        self.descriptors = descriptors

    def getImage(self):
        return self.inputImage

    def getDescriptor(self):
        return self.descriptors

    def getKeypoints(self):
        return self.keypoints

    def onTrackbar(self, value):
        contrastThreshold = value / 100.0
        self.setContrastThreshold(contrastThreshold)

    def saveToCSV(self, filename, keypoints):
        with open(filename, 'w') as csvFile:
            csvFile.write("Index,x,y,size,angle,response\n")
            for i, kp in enumerate(keypoints):
                csvFile.write(f"{i},{kp.pt[0]},{kp.pt[1]},{kp.size},{kp.angle},{kp.response}\n")

    def drawIndexesToImage(self, image, keypoints):
        for i, kp in enumerate(keypoints):
            index = str(i)
            cv2.putText(image, index, (int(kp.pt[0]), int(kp.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


class Matcher:
    def __init__(self, training, validation):
        self.trainingKeypoints = training.getKeypoints()
        self.validationKeypoints = validation.getKeypoints()
        self.trainingDescription = training.getDescriptor()
        self.validationDescription = validation.getDescriptor()
        self.trainingImage = training.getImage()
        self.validationImage = validation.getImage()
        self.matches = []

    def matchFeatures(self):
        self.matches = self.computeMatches(self.trainingDescription, self.validationDescription)
        return self.matches

    def computeMatches(self, trainingDescription, validationDescription):
        bf = cv2.BFMatcher()
        matches = bf.match(trainingDescription, validationDescription)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def filterMatches(self, TH):
        good_matches = [match for match in self.matches if match.distance < TH]
        return good_matches

    def drawMatches(self, matches, imgMatches):
        self.imgMatches = cv2.drawMatches(self.trainingImage, self.trainingKeypoints, 
                                           self.validationImage, self.validationKeypoints, 
                                           matches, imgMatches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



def LoadXYZCoordinates(filePath):
    coordinates = []

    # Datei öffnen
    with open(filePath, 'r') as file:
        # Header-Zeile überspringen
        file.readline()

        # Zeilen einlesen und verarbeiten
        for line in file:
            # Überprüfen, ob die Zeile leer ist
            if not line.strip():
                continue
            
            # CSV-Werte einlesen
            parts = line.strip().split(',')
            if len(parts) != 4:
                print("Ungültiges CSV-Format in Zeile:", line)
                continue

            try:
                index, x, y, z = map(float, parts)
                coordinates.append((x, y, z))
            except ValueError as e:
                print("Fehler beim Parsen der Zeile:", line)
                continue

    return np.array(coordinates, dtype=np.float32)



def estimate_pose(matches, keypoints1, keypoints2, object_points, camera_matrix, dist_coeffs):
    image_points = []
    selected_object_points = []

    for match in matches:
        image_points.append(keypoints2[match.trainIdx].pt)

        index = match.queryIdx
        if index < len(object_points):
            selected_object_points.append(object_points[index])

    if len(selected_object_points) == len(image_points) and len(selected_object_points) >= 4:
        success, rvec, tvec = cv2.solvePnP(np.array(selected_object_points), np.array(image_points), camera_matrix, dist_coeffs)

        if not success:
            print("Pose estimation failed.")
            return None, None

        print("Translation Vector:")
        print(tvec)
        print("Rotation Vector:")
        print(rvec)

        return rvec, tvec
    else:
        print("Number of object points and image points do not match or not enough points for pose estimation.")
        return None, None

# Beispiel-Nutzung des Codes
if __name__ == "__main__":
    trainingImagePath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image.png"
    training = FeatureExtraction(trainingImagePath)
    training.computeKeypointsAndDescriptors()

    videoPath = "/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_video_real.mp4"
    video = cv2.VideoCapture(videoPath)

    activeSet_XYZ_Path = "/home/fhtw_user/catkin_ws/src/HW_3/src/PoseEstimation/activeSet_XYZ.csv"

    # Laden der XYZ-Koordinaten aus der CSV-Datei
    worldPoints = LoadXYZCoordinates(activeSet_XYZ_Path)  # Diese Funktion muss noch implementiert werden

    while True:
        ret, frame = video.read()
        if not ret:
            break

        validation = FeatureExtraction()
        validation.inputImage = frame
        validation.computeKeypointsAndDescriptors()

        matcher = Matcher(training, validation)
        matches = matcher.matchFeatures()
        filteredMatches = matcher.filterMatches(150)

        outputImage = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        matcher.drawMatches(filteredMatches, outputImage)

        # Kameramatrix (intrinsische Parameter)
        cameraMatrix = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32)

        # Verzerrungskoeffizienten
        distCoeffs = np.zeros((4, 1), dtype=np.float32)

        rvec, tvec = estimate_pose(filteredMatches, training.getKeypoints(), validation.getKeypoints(), worldPoints, cameraMatrix, distCoeffs)
        print("Translation Vector:\n", tvec)
        print("Rotation Vector:\n", rvec)

        cv2.imshow("Brute Force Matching", outputImage)

        if cv2.waitKey(int(1000 / video.get(cv2.CAP_PROP_FPS))) == 27:
            break

    cv2.destroyAllWindows()




























# import numpy as np
# import cv2

# # 3D-Punkte im Weltkoordinatensystem
# object_points = np.array([[0, 0, 0],
#                            [1, 0, 0],
#                            [0, 1, 0],
#                            [1, 1, 0]], dtype=np.float32)

# # Ecken des Musters im Bild
# image_points = np.array([[10, 10],
#                          [20, 10],
#                          [10, 20],
#                          [20, 20]], dtype=np.float32)

# # Kameramatrix (intrinsische Parameter)
# camera_matrix = np.array([[100, 0, 50],
#                           [0, 100, 50],
#                           [0, 0, 1]], dtype=np.float32)

# # Verzerrungskoeffizienten
# dist_coeffs = np.zeros((4,1))

# # Laden des Bildes
# image = cv2.imread('/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Mustererkennung (z.B. mit cornerSubPix)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# corners = cv2.cornerSubPix(gray, np.float32(image_points), (11, 11), (-1, -1), criteria)

# # solvePnP aufrufen, um die Pose zu schätzen
# retval, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

# # Projizieren der Achsen des Weltkoordinatensystems auf das Bild
# axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
# imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

# # Zeichnen der Achsen
# image = cv2.line(image, tuple(corners[0].astype(int)), tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
# image = cv2.line(image, tuple(corners[0].astype(int)), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
# image = cv2.line(image, tuple(corners[0].astype(int)), tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)

# # Drucken der Ergebnisse
# print("Rotation vector (rvec):")
# print(rvec)
# print("Translation vector (tvec):")
# print(tvec)

# # Anzeigen des Ergebnisbildes
# cv2.imshow('Image with axis', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

















# import cv2
# import numpy as np

# # Lade zwei Bilder
# img1 = cv2.imread('/home/fhtw_user/catkin_ws/src/HW_3/data/WhatsApp Image 2024-05-02 at 19.56.19.jpeg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png', cv2.IMREAD_GRAYSCALE)

# # Erzeuge einen SIFT-Detektor
# sift = cv2.SIFT_create()

# # Finde die Key-Points und Deskriptoren für beide Bilder
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# # Erzeuge einen Brute-Force-Matcher
# bf = cv2.BFMatcher()

# # Führe eine Brute-Force-Matching durch
# matches = bf.match(descriptors1, descriptors2)

# # Sortiere die Matches nach ihrer Distanz
# matches = sorted(matches, key = lambda x:x.distance)

# # Zeige die ersten 10 Matches
# img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow("Matches", img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
