import numpy as np
import cv2

# 3D-Punkte im Weltkoordinatensystem
object_points = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 1, 0],
                           [1, 1, 0]], dtype=np.float32)

# Ecken des Musters im Bild
image_points = np.array([[10, 10],
                         [20, 10],
                         [10, 20],
                         [20, 20]], dtype=np.float32)

# Kameramatrix (intrinsische Parameter)
camera_matrix = np.array([[100, 0, 50],
                          [0, 100, 50],
                          [0, 0, 1]], dtype=np.float32)

# Verzerrungskoeffizienten
dist_coeffs = np.zeros((4,1))

# Laden des Bildes
image = cv2.imread('/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mustererkennung (z.B. mit cornerSubPix)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(image_points), (11, 11), (-1, -1), criteria)

# solvePnP aufrufen, um die Pose zu schätzen
retval, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

# Projizieren der Achsen des Weltkoordinatensystems auf das Bild
axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

# Zeichnen der Achsen
image = cv2.line(image, tuple(corners[0].astype(int)), tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
image = cv2.line(image, tuple(corners[0].astype(int)), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
image = cv2.line(image, tuple(corners[0].astype(int)), tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)

# Drucken der Ergebnisse
print("Rotation vector (rvec):")
print(rvec)
print("Translation vector (tvec):")
print(tvec)

# Anzeigen des Ergebnisbildes
cv2.imshow('Image with axis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

















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
