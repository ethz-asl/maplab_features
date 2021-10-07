import rospy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import LkConfig

class FeatureDetector(object):
    def __init__(self, config):
        self.config = config
        self.detector = self.init_feature_detector(config)
        self.describer = self.init_feature_describer(config)

    def init_feature_detector(self, config):
        type = config.feature_detector
        if type == 'sift':
            return cv2.xfeatures2d.SIFT_create()
        if type == 'surf':
            surf = cv2.xfeatures2d.SURF_create()
            surf.setHessianThreshold(config.surf_hessian_threshold)
            surf.setNOctaves(config.surf_n_octaves)
            surf.setNOctaveLayers(config.surf_n_octaves_layers)
            return surf
        else:
            rospy.logerr('[FeatureDetector] Unknown feature type: {type_name}'.format(type_name=type))
            return None

    def init_feature_describer(self, config):
        type = config.feature_descriptor
        if type == 'freak':
            return cv2.xfeatures2d.FREAK_create()
        elif type == 'brief':
            return cv2.xfeatures2d.BRIEF_create()
        elif type == 'sift':
            return cv2.xfeatures2d.SIFT_create()
        elif type == 'surf':
            surf = cv2.xfeatures2d.SURF_create()
            surf.setHessianThreshold(config.surf_hessian_threshold)
            surf.setNOctaves(config.surf_n_octaves)
            surf.setNOctaveLayers(config.surf_n_octaves_layers)
            return surf
        else:
            rospy.logerr('[FeatureDetector] Unknown feature type: {type_name}'.format(type_name=type))
            return None

    def detect_and_describe(self, cv_img):
        if self.detector is None or self.describer is None:
            rospy.logerr('[FeatureDetector] Cannot detect features. Initializing failed mostly likely due to a bad configuration.')
            return np.array([]), np.array([])

        keypoints = self.detector.detect(cv_img, None)
        keypoints, descriptors = self.describer.compute(cv_img, keypoints)

        # Display keypoints on image.
        if self.config.debug_feature_extraction:
            img = cv2.drawKeypoints(cv_img, keypoints, None, (255,0,0), 4)
            plt.imshow(img)
            plt.show()
        return self.cv_keypoints_to_features(keypoints, descriptors)

    def cv_keypoints_to_features(self, keypoints, descriptors):
        n_keypoints = len(keypoints)
        assert n_keypoints == descriptors.shape[0]
        features = np.zeros((n_keypoints, 4 + descriptors.shape[1]))
        for i in range(n_keypoints):
            features[i, :2] = [keypoints[i].pt[0], keypoints[i].pt[1]] # or 1,0?
            features[i, 2] = keypoints[i].response
            features[i, 3] = keypoints[i].octave
            features[i, 4:] = descriptors[i,:]
        return features


if __name__ == '__main__':
    config = LkConfig()
    config.feature_detector = 'surf'
    config.feature_descriptor = 'surf'
    config.debug_feature_extraction = True
    config.surf_hessian_threshold = 50000

    fd = FeatureDetector(config)
    img = cv2.imread('../share/fly.png',0)

    # Extract keypoints and descriptors from the image.
    features = fd.detect_and_describe(img)

    # Display keypoints on image.
    # img2 = cv2.drawKeypoints(img, keypoints, None, (255,0,0), 4)
    # plt.imshow(img2)
    # plt.show()

    print(features.shape)
