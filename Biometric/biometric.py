import cv2
import numpy as np
import os

def biometrics(img_path):
    path = "Biometric/database/"

    test_original = cv2.imread(img_path)

    for file in [file for file in os.listdir(path)]:
        
        fingerprint_database_image = cv2.imread(path+file)
        
        sift = cv2.SIFT_create()
        
        keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)
        
        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),dict()).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []
    
        for p, q in matches:
            if p.distance < 0.1*q.distance:
                match_points.append(p)

        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            keypoints = len(keypoints_1)            
        else:
            keypoints = len(keypoints_2)

        if (len(match_points) / keypoints)>0.95:
            result = str(file).split('.')[0]
            return result
    
    return 'none'
    