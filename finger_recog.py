import cv2
import fingerprint_feature_extractor

img = cv2.imread('enhanced/1.jpg', 0)

FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
    img, 
    spuriousMinutiaeThresh=10, 
    invertImage=False, 
    showResult=True, 
    saveResult=True
    )
