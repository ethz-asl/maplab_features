import os
import sys
import cv2
import numpy as np
import threading
import torch

module_path = os.path.join(os.path.dirname(__file__), 'trackers/superglue')
if module_path not in sys.path:
    sys.path.append(module_path)

from models.superpoint import SuperPoint

def visualize_detections(frame, xy):
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for kp in xy:
        cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
    cv2.imshow("Detections", vis)
    cv2.waitKey(3)

class FeatureExtractionSuperPoint(object):
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()

        # Network related initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.superpoint = SuperPoint({
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }).eval().to(self.device)

    def frame2tensor(self, frame, device):
        return torch.from_numpy(frame/255.).float()[None, None].to(device)

    def detect_and_describe(self, cv_image):
        # Lock thread for detection duration
        self.lock.acquire()

        with torch.set_grad_enabled(False):
            # Preprocess for pytorch
            torch_image = self.frame2tensor(
                cv_image.astype(np.float32), self.device)

            # Get superpoint output
            pred = self.superpoint({'image': torch_image})
            xy = pred['keypoints'][0].cpu().numpy().astype(np.float32)
            scores = pred['scores'][0].cpu().numpy().astype(np.float32)
            scales = np.zeros(xy.shape[0]).astype(np.float32)
            descriptors = pred['descriptors'][0].cpu().numpy()
            descriptors = descriptors.transpose((1, 0)).astype(np.float32)

        self.lock.release()

        if self.config.debug_detections:
            visualize_detections(cv_image, xy)

        return xy, scores, scales, descriptors

class FeatureExtractionCv(object):
    def __init__(self, config):
        self.config = config
        self.detector = self.init_feature_detector(config)
        self.describer = self.init_feature_describer(config)

    def init_feature_detector(self, config):
        type = config.cv_feature_detector
        if type == 'sift':
            return cv2.SIFT_create()
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
            return cv2.SIFT_create()
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
            visualize_detections(cv_img, xy)

        return xy, scores, scales, descriptors

    def cv_keypoints_to_features(self, cv_keypoints, cv_descriptors):
        n_keypoints = len(cv_keypoints)
        if n_keypoints == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        assert n_keypoints == cv_descriptors.shape[0]

        xy = np.zeros((n_keypoints, 2))
        scores = np.zeros((n_keypoints,), dtype=np.float32)
        scales = np.zeros((n_keypoints,), dtype=np.float32)
        descriptors = np.zeros(
            (n_keypoints, cv_descriptors.shape[1]), dtype=np.float32)
        for i in range(n_keypoints):
            xy[i] = [cv_keypoints[i].pt[0], cv_keypoints[i].pt[1]]
            scores[i] = cv_keypoints[i].response
            scales[i] = cv_keypoints[i].octave
            descriptors[i] = cv_descriptors[i,:]
        return xy, scores, scales, descriptors
