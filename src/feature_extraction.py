import rospy
import cv2
import numpy as np

from utils_py2 import open_fifo, read_np, send_np, read_bytes

class FeatureExtractionExternal(object):
    def __init__(self, config):
        self.config = config

    def detect_and_describe(self, cv_img):
        # Transmit image for processing
        cv_success, cv_binary = cv2.imencode('.png', cv_img)
        assert(cv_success)
        send_np(self.config.fifo_features_out, cv_binary)

        xy = read_np(self.config.fifo_features_in, np.float32)
        scores = read_np(self.config.fifo_features_in, np.float32)
        scales = read_np(self.config.fifo_features_in, np.float32)
        descriptors = read_np(self.config.fifo_features_in, np.float32)

        return xy, scores, scales, descriptors

class FeatureExtractionCv(object):
    def __init__(self, config):
        self.config = config
        self.detector = self.init_feature_detector(config)
        self.describer = self.init_feature_describer(config)

    def init_feature_detector(self, config):
        type = config.cv_feature_detector
        if type == 'sift':
            return cv2.xfeatures2d.SIFT_create()
        if type == 'surf':
            surf = cv2.xfeatures2d.SURF_create()
            surf.setHessianThreshold(config.surf_hessian_threshold)
            surf.setNOctaves(config.surf_n_octaves)
            surf.setNOctaveLayers(config.surf_n_octaves_layers)
            return surf
        else:
            raise ValueError(
                '[FeatureDetector] Unknown feature type: {type_name}'.format(
                    type_name=type))

    def init_feature_describer(self, config):
        type = config.cv_feature_descriptor
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
            raise ValueError(
                '[FeatureDetector] Unknown feature type: {type_name}'.format(
                    type_name=type))

    def detect_and_describe(self, cv_img):
        if self.detector is None or self.describer is None:
            raise ValueError(
                '''[FeatureDetector] Cannot detect features. Initializing
                failed mostly likely due to a bad configuration.''')

        keypoints = self.detector.detect(cv_img, None)
        keypoints, descriptors = self.describer.compute(cv_img, keypoints)
        return self.cv_keypoints_to_features(keypoints, descriptors)

    def cv_keypoints_to_features(self, keypoints, descriptors):
        n_keypoints = len(keypoints)
        if n_keypoints == 0:
            return np.array([])
        assert n_keypoints == descriptors.shape[0]

        xy = np.zeros((n_keypoints, 2))
        scores = np.zeros((n_keypoints,))
        scales = np.zeros((n_keypoints,))
        descriptors = np.zeros((n_keypoints, descriptors.shape[1]))
        for i in range(n_keypoints):
            xy[i] = [keypoints[i].pt[0], keypoints[i].pt[1]]
            scores[i] = keypoints[i].response
            scales[i] = keypoints[i].octave
            descriptors[i] = descriptors[i,:]
        return xy, scores, scales, descriptors
