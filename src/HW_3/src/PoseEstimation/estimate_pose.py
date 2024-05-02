import cv2
import numpy as np

# Lade zwei Bilder
img1 = cv2.imread('/home/fhtw_user/catkin_ws/src/HW_3/data/WhatsApp Image 2024-05-02 at 19.56.19.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png', cv2.IMREAD_GRAYSCALE)

# Erzeuge einen SIFT-Detektor
sift = cv2.SIFT_create()

# Finde die Key-Points und Deskriptoren für beide Bilder
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Erzeuge einen Brute-Force-Matcher
bf = cv2.BFMatcher()

# Führe eine Brute-Force-Matching durch
matches = bf.match(descriptors1, descriptors2)

# Sortiere die Matches nach ihrer Distanz
matches = sorted(matches, key = lambda x:x.distance)

# Zeige die ersten 10 Matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
