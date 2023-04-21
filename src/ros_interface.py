#!/usr/bin/env python3

import os
import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue

import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from maplab_msgs.msg import Features
from std_msgs.msg import MultiArrayDimension

from config import MainConfig
from feature_extraction import FeatureExtractionCv, FeatureExtractionSuperPoint
from feature_tracking import FeatureTrackingLK, FeatureTrackingSuperGlue

class ImageReceiver(Thread):
    def __init__(self, config, feature_extractor, feature_tracker, index):
        Thread.__init__(self, args=(), daemon=True)

        self.config = config
        self.index = index
        self.feature_extractor = feature_extractor
        self.feature_tracker = feature_tracker

        # Image subscriber
        self.image_queue = Queue()
        self.image_sub = rospy.Subscriber(
            self.config.input_topic[self.index], Image,
            self.image_callback, queue_size=10)
        self.descriptor_pub = rospy.Publisher(
            self.config.output_topic[self.index], Features,
            queue_size=10)
        rospy.loginfo('[ImageReceiver] Subscribed to {in_topic}'.format(
                in_topic=self.config.input_topic[self.index]) +
            ' and publishing to {out_topic}'.format(
                out_topic=self.config.output_topic[self.index]))

        # CV bridge for conversion
        self.bridge = CvBridge()

        # Data on the last processed frame
        self.prev_xy = []
        self.prev_scales = []
        self.prev_scores = []
        self.prev_descriptors = []
        self.prev_track_ids = []
        self.prev_frame = []

        # Initialize internal counters
        self.image_count = 0
        self.next_track_id = 0

    def detect_and_describe(self, cv_image):
        # Get keypoints and descriptors.
        self.xy, self.scores, self.scales, self.descriptors = \
            self.feature_extractor.detect_and_describe(cv_image)

        if len(self.xy) > 0:
            # Do not detect next to the image border
            img_h, img_w = cv_image.shape[:2]
            top_and_left = np.logical_and(
                self.xy[:, 0] > self.config.min_distance_to_image_border,
                self.xy[:, 1] > self.config.min_distance_to_image_border)
            bot_and_right = np.logical_and(
                self.xy[:, 0] < img_w - self.config.min_distance_to_image_border,
                self.xy[:, 1] < img_h - self.config.min_distance_to_image_border)
            keep = np.logical_and(top_and_left, bot_and_right)

            self.xy = self.xy[keep, :2]
            self.scores = self.scores[keep]
            self.scales = self.scales[keep]
            self.descriptors = self.descriptors[keep]

    def initialize_new_tracks(self, cv_image):
        if len(self.prev_xy) > 0:
            # Limit number of new detections added to fit with the global limit,
            # and mask detections to not initialize keypoints that are too close
            # to previous ones or to new ones
            quota = self.config.num_tracked_points - self.prev_xy.shape[0]
            mask = np.ones((cv_image.shape[0], cv_image.shape[1]))
            for kp in self.prev_xy:
                x, y = kp.astype(np.int32)
                cv2.circle(mask, (x, y), self.config.mask_redetections_thr_px,
                    0, cv2.FILLED)

            keep = []
            for i in range(self.xy.shape[0]):
                if quota <= 0:
                    break
                x, y = self.xy[i].astype(np.int32)
                if mask[y, x]:
                    keep.append(i)
                    quota -= 1
            keep = np.array(keep)

            # Assign new track ids
            if keep.size > 0:
                track_ids = np.arange(
                    self.next_track_id,
                    self.next_track_id + keep.size).astype(np.int32)
                self.next_track_id += keep.size

                self.prev_xy = np.concatenate([self.prev_xy, self.xy[keep]])
                self.prev_scores = np.concatenate(
                    [self.prev_scores, self.scores[keep]])
                self.prev_scales = np.concatenate(
                    [self.prev_scales, self.scales[keep]])
                self.prev_descriptors = np.concatenate(
                    [self.prev_descriptors, self.descriptors[keep]])
                self.prev_track_ids = np.concatenate(
                    [self.prev_track_ids, track_ids])
        else:
            # If there are no previous keypoints no need for complicated logic
            num_new_keypoints = min(
                self.xy.shape[0], self.config.num_tracked_points)
            self.prev_xy = self.xy[:num_new_keypoints].copy()
            self.prev_scores = self.scores[:num_new_keypoints].copy()
            self.prev_scales = self.scales[:num_new_keypoints].copy()
            self.prev_descriptors = self.descriptors[:num_new_keypoints].copy()

            # Assign new track ids
            self.prev_track_ids = np.arange(
                self.next_track_id,
                self.next_track_id + num_new_keypoints).astype(np.int32)
            self.next_track_id += num_new_keypoints

    def publish_features(self, stamp):
        num_keypoints = int(self.prev_xy.shape[0])
        descriptors = self.prev_descriptors.astype(np.float32)

        # If available PCA descriptor before exporting
        if self.config.pca_descriptors:
            descriptors = self.pca.transform(descriptors)

        # Flatten descriptors and convert to bytes
        descriptors = descriptors.flatten().view(np.uint8)

        # Fill in basic message data
        feature_msg = Features()
        feature_msg.header.stamp = stamp
        feature_msg.numKeypointMeasurements = num_keypoints
        feature_msg.keypointMeasurementsX = (
            self.prev_xy[:, 0] / self.config.resize_input_image).tolist()
        feature_msg.keypointMeasurementsY = (
            self.prev_xy[:, 1] / self.config.resize_input_image).tolist()
        feature_msg.keypointMeasurementUncertainties = [0.8] * num_keypoints
        feature_msg.keypointScales = self.prev_scales.tolist()
        feature_msg.keypointScores = self.prev_scores.tolist()
        feature_msg.descriptors.data = descriptors.tolist()
        feature_msg.trackIds = self.prev_track_ids.tolist()

        # Descriptor array dimentions
        dim0 = MultiArrayDimension()
        dim0.label = 'desc_count'
        dim0.size = int(num_keypoints)
        dim0.stride = int(descriptors.size)

        dim1 = MultiArrayDimension()
        dim1.label = 'desc_size'
        desc_bytes =  int(descriptors.size / num_keypoints)
        assert(desc_bytes * num_keypoints == descriptors.size)
        dim1.size = desc_bytes
        dim1.stride = desc_bytes

        feature_msg.descriptors.layout.dim = [dim0, dim1]
        feature_msg.descriptors.layout.data_offset = 0

        # Publish
        self.descriptor_pub.publish(feature_msg)

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                    image_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        if self.config.resize_input_image != 1.0:
            h, w = cv_image.shape[:2]
            nh = int(h * self.config.resize_input_image)
            nw = int(w * self.config.resize_input_image)
            cv_image = cv2.resize(cv_image, (nw, nh))

        timestamp = image_msg.header.stamp
        self.image_queue.put((timestamp, cv_image))

    def run(self):
        while True:
            # Wait for next image
            while self.image_queue.empty():
                time.sleep(0.01)

            timestamp, cv_image = self.image_queue.get()

            # Get both color and grayscale versions for the image
            if cv_image.ndim == 3 and cv_image.shape[2] == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Find keypoints and extract features
            self.detect_and_describe(cv_image)

            # If we have a previous frame and features track them
            if len(self.prev_xy) > 0 and len(self.prev_frame) > 0:
                self.prev_xy, self.prev_scores, self.prev_scales, \
                self.prev_descriptors, self.prev_track_ids = \
                    self.feature_tracker.track(
                        self.prev_frame, cv_image,
                        self.prev_xy, self.xy,
                        self.prev_scores, self.scores,
                        self.prev_scales, self.scales,
                        self.prev_descriptors, self.descriptors,
                        self.prev_track_ids)
            self.prev_frame = cv_image

            if len(self.xy) > 0:
                self.initialize_new_tracks(cv_image)

            self.image_count += 1
            if self.image_count % 10 == 0:
                print('topic {index} recv {count}'.format(
                    index=self.index, count=self.image_count))

            if len(self.prev_xy) > 0:
                self.publish_features(timestamp)

if __name__ == '__main__':
    rospy.init_node('maplab_features', anonymous=True)

    config = MainConfig()
    config.init_from_config()

    # Initialize shared feature extraction and tracking
    if config.feature_extraction == 'cv':
        feature_extractor = FeatureExtractionCv(config)
    elif config.feature_extraction == 'superpoint':
        feature_extractor = FeatureExtractionSuperPoint(config)
    else:
        raise ValueError('Invalid feature extraction type: {feature}'.format(
            feature=config.feature_extraction))

    if config.feature_tracking == 'lk':
        feature_tracker = FeatureTrackingLK(config)
    elif config.feature_tracking == 'superglue':
        feature_tracker = FeatureTrackingSuperGlue(config)
    else:
        raise ValueError('Invalid feature tracking method: {tracker}'.format(
            tracker=config.feature_tracking))

    # Feature compression with PCA
    if config.pca_descriptors:
        import pickle
        with open(config.pca_pickle_path, 'rb') as pickle_file:
            pca = pickle.load(pickle_file)
        rospy.loginfo(
            '[ImageReceiver] Using PCA to project from ' +
            '{:d} to {:d} feature size.'.format(
                pca.n_features_ , pca.n_components_))

    receives = []
    for i in range(len(config.input_topic)):
        receiver = ImageReceiver(
            config, feature_extractor, feature_tracker, i)
        receiver.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")
