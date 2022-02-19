#!/usr/bin/env python
from __future__ import print_function

import cv2
import numpy as np

import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from maplab_msgs.msg import Features
from std_msgs.msg import MultiArrayDimension

from config import LkConfig
from utils_py2 import open_fifo, read_np, send_np
from feature_extraction import FeatureExtractionCv, FeatureExtractionExternal
from feature_tracking import FeatureTrackingLK, FeatureTrackingExternal

class ImageReceiver:
    def __init__(self):
        self.config = LkConfig()
        self.config.init_from_config()

        # Image subscriber
        self.image_sub = rospy.Subscriber(
                self.config.input_topic, Image, self.image_callback, queue_size=4000)
        self.descriptor_pub = rospy.Publisher(
                self.config.output_topic, Features, queue_size=100)
        rospy.loginfo('[ImageReceiver] Subscribed to {in_topic} and ' +
            'publishing to {out_topic}'.format(
                in_topic=self.config.input_topic,
                out_topic=self.config.output_topic))

        # CV bridge for conversion
        self.bridge = CvBridge()

        # Feature detection and description
        if self.config.feature_extraction == 'cv':
            self.feature_extraction = FeatureExtractionCv(self.config)
        elif self.config.feature_extraction == 'external':
            self.feature_extraction = FeatureExtractionExternal(self.config)
        else:
            ValueError('Invalid feature extraction type: {feature}'.format(
                feature=self.config.feature_extraction))

        self.tracker = 'superglue'

        if self.tracker == 'lk':
            self.feature_tracking = FeatureTrackingLK(self.config)
        elif self.tracker == 'superglue':
            self.feature_tracking = FeatureTrackingExternal(self.config)
        else:
            ValueError('Invalid feature tracking method: {tracker}'.format(
                tracker=self.tracker))

        self.num_keypoints = 600
        self.resize_input_image = 640
        self.min_distance_to_image_border = 30
        self.mask_redetections_thr_px = 7

        self.debug = True
        self.count_recv = 0

        # Data on the last processed frame
        self.prev_xy = []
        self.prev_scales = []
        self.prev_scores = []
        self.prev_descriptors = []
        self.prev_track_ids = []
        self.prev_frame = []

        self.next_track_id = 0

    def detect_and_describe(self, cv_image):
        # Get keypoints and descriptors.
        self.xy, self.scores, self.scales, self.descriptors = \
            self.feature_extraction.detect_and_describe(cv_image)

        # Do not detect next to the image border
        img_h, img_w = cv_image.shape[:2]
        top_and_left = np.logical_and(
            self.xy[:, 0] > self.min_distance_to_image_border,
            self.xy[:, 1] > self.min_distance_to_image_border)
        bot_and_right = np.logical_and(
            self.xy[:, 0] < img_w - self.min_distance_to_image_border,
            self.xy[:, 1] < img_h - self.min_distance_to_image_border)
        keep = np.logical_and(top_and_left, bot_and_right)

        self.xy = self.xy[keep, :2]
        self.scores = self.scores[keep]
        self.scales = self.scales[keep]
        self.descriptors = self.descriptors[keep]

        if self.debug:
            vis = cv_image.copy()
            for kp in self.xy:
                cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
            cv2.imshow("Detections", vis)
            cv2.waitKey(3)

    def publish_features(self, stamp):
        num_keypoints = int(self.prev_xy.shape[0])

        # Flatten descriptors and convert to bytes
        descriptors = self.prev_descriptors.flatten().view(np.uint8)

        # Fill in basic message data
        feature_msg = Features()
        feature_msg.header.stamp = stamp
        feature_msg.numKeypointMeasurements = num_keypoints
        feature_msg.keypointMeasurementsX = (self.prev_xy[:, 0] / self.scale).tolist()
        feature_msg.keypointMeasurementsY = (self.prev_xy[:, 1] / self.scale).tolist()
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

        self.count_recv += 1

        if self.resize_input_image != -1:
            h, w = cv_image.shape[:2]
            self.scale = self.resize_input_image / float(max(h, w))
            nh, nw = int(h * self.scale), int(w * self.scale)
            cv_image = cv2.resize(cv_image, (nw, nh))

        if cv_image.ndim == 3 and cv_image.shape[2] == 3:
            frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            frame_color = cv_image
        else:
            frame_gray = cv_image
            frame_color = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        # Find keypoints and extract features
        self.detect_and_describe(frame_color)

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

        if len(self.prev_xy) > 0:
            # Limit number of new detections added to fit with the global limit,
            # and mask detections to not initialize keypoints that are too close
            # to previous ones or to new ones
            quota = self.num_keypoints - self.prev_xy.shape[0]
            mask = np.ones((cv_image.shape[0], cv_image.shape[1]))
            for kp in self.prev_xy:
                x, y = kp.astype(np.int32)
                cv2.circle(mask, (x, y), self.mask_redetections_thr_px, 0,
                    cv2.FILLED)

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
                self.prev_scores = np.concatenate([self.prev_scores, self.scores[keep]])
                self.prev_scales = np.concatenate([self.prev_scales, self.scales[keep]])
                self.prev_descriptors = np.concatenate(
                    [self.prev_descriptors, self.descriptors[keep]])
                self.prev_track_ids = np.concatenate(
                    [self.prev_track_ids, track_ids])
        else:
            # If there are no previous keypoints no need for complicated logic
            num_new_keypoints = min(self.xy.shape[0], self.num_keypoints)
            self.prev_xy = self.xy[:num_new_keypoints].copy()
            self.prev_scores = self.scores[:num_new_keypoints].copy()
            self.prev_scales = self.scales[:num_new_keypoints].copy()
            self.prev_descriptors = self.descriptors[:num_new_keypoints].copy()

            # Assign new track ids
            self.prev_track_ids = np.arange(
                self.next_track_id,
                self.next_track_id + num_new_keypoints).astype(np.int32)
            self.next_track_id += num_new_keypoints

        print('received', self.count_recv)

        if len(self.prev_xy) > 0:
            self.publish_features(image_msg.header.stamp)

if __name__ == '__main__':
    rospy.init_node('lk_tracker', anonymous=True)
    receiver = ImageReceiver()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")
