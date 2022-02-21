import rospy
import cv2
import numpy as np

from utils_py2 import open_fifo, read_np, send_np, read_bytes

def visualize_detections(frame, xy, index):
    vis = frame.copy()
    for kp in xy:
        cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
    cv2.imshow("Detections {index}".format(index=index), vis)
    cv2.waitKey(3)

class FeatureExtractionExternal(object):
    def __init__(self, config, index):
        self.config = config
        self.index = index

    def detect_and_describe(self, cv_img):
        # Encode image.
        cv_success, cv_binary = cv2.imencode('.png', cv_img)
        assert(cv_success)

        # Lock thread for transmission duration.
        self.config.lock_features.acquire()

        # Send image to external module.
        send_np(self.config.fifo_features_out, cv_binary)

        # Receive detected features.
        xy = read_np(self.config.fifo_features_in, np.float32)
        scores = read_np(self.config.fifo_features_in, np.float32)
        scales = read_np(self.config.fifo_features_in, np.float32)
        descriptors = read_np(self.config.fifo_features_in, np.float32)

        if self.config.debug_detections:
            visualize_detections(cv_img, xy, self.index)

        self.config.lock_features.release()

        return xy, scores, scales, descriptors

class FeatureExtractionCv(object):
    def __init__(self, config, index):
        self.config = config
        self.detector = self.init_feature_detector(config)
        self.describer = self.init_feature_describer(config)
        self.index = index

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
        xy, scores, scales, descriptors = self.cv_keypoints_to_features(
            keypoints, descriptors)

        if self.config.debug_detections:
            self.config.lock_features.acquire()
            visualize_detections(cv_img, xy, self.index)
            self.config.lock_features.release()

        return xy, scores, scales, descriptors

    def cv_keypoints_to_features(self, keypoints, descriptors):
        n_keypoints = len(keypoints)
        if n_keypoints == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
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
