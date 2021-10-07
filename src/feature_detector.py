import rospy
import cv2
import matplotlib.pyplot as plt
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
        return keypoints, descriptors

if __name__ == '__main__':
    config = LkConfig()
    config.feature_detector = 'surf'
    config.feature_descriptor = 'surf'
    config.surf_hessian_threshold = 50000
    
    fd = FeatureDetector(config)
    img = cv2.imread('../share/fly.png',0)

    # Extract keypoints and descriptors from the image.
    keypoints, descriptors = fd.detect_and_describe(img)

    # Display keypoints on image.
    img2 = cv2.drawKeypoints(img, keypoints, None, (255,0,0), 4)
    plt.imshow(img2)
    plt.show()
