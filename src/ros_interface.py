#!/usr/bin/env python
from __future__ import print_function

import cv2
import numpy as np
import threading
import os

import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from maplab_msgs.msg import Features
from std_msgs.msg import MultiArrayDimension

from config import MainConfig
from utils_py2 import open_fifo, read_np, send_np
from feature_extraction import FeatureExtractionCv, FeatureExtractionExternal
from feature_tracking import FeatureTrackingLK, FeatureTrackingExternal

class ImageReceiver:
    def __init__(self, config, index):
        self.config = config
        self.index = index

        # Image subscriber
        self.image_sub = rospy.Subscriber(
            self.config.input_topic[self.index], Image,
            self.image_callback, queue_size=4000)
        self.descriptor_pub = rospy.Publisher(
            self.config.output_topic[self.index], Features,
            queue_size=100)
        rospy.loginfo('[ImageReceiver] Subscribed to {in_topic}'.format(
                in_topic=self.config.input_topic[self.index]) +
            ' and publishing to {out_topic}'.format(
                out_topic=self.config.output_topic[self.index]))

        # If masks are available subscribe to that topic as well
        if len(self.config.mask_topic) > 0:
            self.mask_sub = rospy.Subscriber(
                self.config.mask_topic[self.index], Image,
                self.mask_callback, queue_size=4000)
            self.mask_buffer = {}
            self.mask_buffer_mutex = threading.Lock()
        else:
            self.mask_buffer = None

        # CV bridge for conversion
        self.bridge = CvBridge()

        # Feature detection and description
        if self.config.feature_extraction == 'cv':
            self.feature_extraction = FeatureExtractionCv(
                self.config, self.index)
        elif self.config.feature_extraction == 'external':
            self.feature_extraction = FeatureExtractionExternal(
                self.config, self.index)
        else:
            raise ValueError('Invalid feature extraction type: {feature}'.format(
                feature=self.config.feature_extraction))

        if self.config.feature_tracking == 'lk':
            self.feature_tracking = FeatureTrackingLK(
                self.config, self.index)
        elif self.config.feature_tracking == 'superglue':
            self.feature_tracking = FeatureTrackingExternal(
                self.config, self.index)
        else:
            raise ValueError('Invalid feature tracking method: {tracker}'.format(
                tracker=self.config.feature_tracking))

        # Feature compression with PCA
        if self.config.pca_descriptors:
            import pickle
            with open(self.config.pca_pickle_path, 'rb') as pickle_file:
                self.pca = pickle.load(pickle_file)
            rospy.loginfo(
                '[ImageReceiver] Using PCA to project from ' +
                '{:d} to {:d} feature size.'.format(
                    self.pca.n_features_ , self.pca.n_components_))

        # Data on the last processed frame
        self.prev_xy = []
        self.prev_scales = []
        self.prev_scores = []
        self.prev_descriptors = []
        self.prev_track_ids = []
        self.prev_frame = []

        # Initialize internal counters
        self.count_received_images = 0
        self.next_track_id = 0


    def detect_and_describe(self, cv_image, mask=None):
        # Get keypoints and descriptors.
        self.xy, self.scores, self.scales, self.descriptors = \
            self.feature_extraction.detect_and_describe(cv_image)

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

            if mask is not None:
                to_mask = mask[
                    np.rint(self.xy[:, 1]).astype(np.int32),
                    np.rint(self.xy[:, 0]).astype(np.int32)]
                keep = np.logical_and(keep, np.logical_not(to_mask))

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
        descriptors = self.prev_descriptors

        # If available PCA descriptor before exporting
        if not self.pca is None:
            descriptors = self.pca.transform(descriptors)

        # Flatten descriptors and convert to bytes
        descriptors = descriptors.flatten().view(np.uint8)

        # Fill in basic message data
        feature_msg = Features()
        feature_msg.header.stamp = stamp
        feature_msg.numKeypointMeasurements = num_keypoints
        feature_msg.keypointMeasurementsX = (
            self.prev_xy[:, 0] / self.scale).tolist()
        feature_msg.keypointMeasurementsY = (
            self.prev_xy[:, 1] / self.scale).tolist()
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

    def mask_callback(self, mask_msg):
        try:
            cv_mask = self.bridge.imgmsg_to_cv2(
                    mask_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        self.mask_buffer_mutex.acquire()
        self.mask_buffer[mask_msg.header.stamp] = cv_mask
        self.mask_buffer_mutex.release()

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                    image_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        self.count_received_images += 1

        if self.config.resize_input_image != -1:
            h, w = cv_image.shape[:2]
            self.scale = self.config.resize_input_image / float(max(h, w))
            nh, nw = int(h * self.scale), int(w * self.scale)
            cv_image = cv2.resize(cv_image, (nw, nh))
        else:
            self.scale = 1

        if cv_image.ndim == 3 and cv_image.shape[2] == 3:
            frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            frame_color = cv_image
        else:
            frame_gray = cv_image
            frame_color = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        # If we are using masks we have to wait for the corresponding mask
        if self.mask_buffer != None:
            counter = 0
            while image_msg.header.stamp not in self.mask_buffer:
                rospy.sleep(0.001)
                counter += 1
                if counter > 1000:
                    rospy.logerr("[ImageReceiver] Masks should be available " +
                        "but none has arrived for 1 second. Giving up on " +
                        "image at " + str(image_msg.header.stamp))
                    return

            self.mask_buffer_mutex.acquire()
            mask = self.mask_buffer[image_msg.header.stamp]
            del self.mask_buffer[image_msg.header.stamp]
            self.mask_buffer_mutex.release()
        else:
            mask = None

        # Find keypoints and extract features
        self.detect_and_describe(frame_color, mask)

        if len(self.prev_xy) > 0 and len(self.prev_frame) > 0:
            self.prev_xy, self.prev_scores, self.prev_scales, \
            self.prev_descriptors, self.prev_track_ids = \
                self.feature_tracking.track(
                    self.prev_frame, frame_gray,
                    self.prev_xy, self.xy,
                    self.prev_scores, self.scores,
                    self.prev_scales, self.scales,
                    self.prev_descriptors, self.descriptors,
                    self.prev_track_ids)
        self.prev_frame = frame_gray

        if len(self.xy) > 0:
            self.initialize_new_tracks(frame_gray)

        if self.count_received_images % 10 == 0:
            print('topic {index} recv {count}'.format(
                index=self.index, count=self.count_received_images))

        if len(self.prev_xy) > 0:
            self.publish_features(image_msg.header.stamp)

if __name__ == '__main__':
    if not os.path.exists('/tmp/maplab_features'):
        os.makedirs('/tmp/maplab_features')

    rospy.init_node('maplab_features', anonymous=True)

    config = MainConfig()
    config.init_from_config()

    # Initialize pipes for external transfer
    if config.feature_extraction == 'external':
        config.fifo_features_out = open_fifo(
            '/tmp/maplab_features/maplab_features_images', 'wb')
        config.fifo_features_in = open_fifo(
            '/tmp/maplab_features/maplab_features_descriptors', 'rb')
    config.lock_features = threading.Lock()

    if config.feature_tracking == 'superglue':
        config.fifo_tracking_out = open_fifo(
            '/tmp/maplab_features/maplab_tracking_images', 'wb')
        config.fifo_tracking_in = open_fifo(
            '/tmp/maplab_features/maplab_tracking_matches', 'rb')
    config.lock_tracking = threading.Lock()

    receivers = []
    for i in range(len(config.input_topic)):
        receivers.append(ImageReceiver(config, i))

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")
