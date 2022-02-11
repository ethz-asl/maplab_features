#!/usr/bin/env python
from __future__ import print_function

import os
import errno
import sys
import cv2
import numpy as np
from scipy import spatial

import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from maplab_msgs.msg import Features
from std_msgs.msg import MultiArrayDimension
from config import LkConfig
from feature_extraction import FeatureExtractionCv, FeatureExtractionExternal

class ImageReceiver:
    def __init__(self):
        self.config = LkConfig()
        self.config.init_from_config()

        # Image subscriber
        self.image_sub = rospy.Subscriber(
                self.config.input_topic, Image, self.image_callback, queue_size=20)
        self.descriptor_pub = rospy.Publisher(
                self.config.output_topic, Features, queue_size=20)
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

        # LK tracker settings
        self.lk_params = dict(
            winSize  = (15, 15),
            maxLevel = 2,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_max_step_px = 1.0
        self.lk_num_keypoints = 600
        self.lk_redetect_thr = 0.9
        self.lk_merge_tracks_thr_px = 3
        self.lk_mask_redetections_thr_px = 10
        self.descriptor_reassociation_thr = 10

        # Data on the last processed frame
        self.prev_xy = []
        self.prev_scales = []
        self.prev_scores = []
        self.prev_descriptors = []
        self.prev_track_ids = []

        self.next_track_id = 0

    def lk_track(self, frame_color):
        if frame_color.ndim == 3 and frame_color.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_color
            frame_color = cv2.cvtColor(frame_color, cv2.COLOR_GRAY2BGR)

        # Check if there is anything to track
        if len(self.prev_xy) > 0:
            # Use optical from to determine keypoint movement
            p0 = self.prev_xy.reshape(-1, 1, 2).astype(np.float32)
            p1, _, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray, frame_gray, p0, None, **self.lk_params)
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(
                frame_gray, self.prev_frame_gray, p1, None, **self.lk_params)

            # Determine quality based on:
            #   1) distance between the optical flow predictions
            #   2) eliminate tracks that have converged to the same point
            #      (prefer keeping smaller track ids which means the track is
            #      is longer)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            mask = np.ones((frame_gray.shape[0], frame_gray.shape[1]))
            idxs = np.argsort(self.prev_track_ids)

            keep = []
            for i in idxs:
                if d[i] >= self.lk_max_step_px:
                    continue

                x, y = self.prev_xy[i].astype(np.int32)
                if (y < 0 or y >= frame_gray.shape[0] or
                    x < 0 or x >= frame_gray.shape[1]):
                    continue

                if mask[y, x]:
                    keep.append(i)
                    cv2.circle(mask, (x, y), self.lk_merge_tracks_thr_px, 0,
                        cv2.FILLED)
            keep = np.array(keep)

            # Visualization
            old_xy = self.prev_xy[keep].astype(np.int)
            new_xy = p1[keep].reshape(-1, 2).astype(np.int)
            vis = frame_color.copy()
            for xy0, xy1 in zip(old_xy, new_xy):
                cv2.line(vis, tuple(xy0), tuple(xy1), (0, 255, 0), 1)

            cv2.imshow("Tracking", vis)
            cv2.waitKey(3)

            # Drop bad keypoints
            self.prev_xy = p1[keep].reshape(-1, 2)
            self.prev_scales = self.prev_scales[keep]
            self.prev_scores = self.prev_scores[keep]
            self.prev_descriptors = self.prev_descriptors[keep]
            self.prev_track_ids = self.prev_track_ids[keep]

        # Save current frame for next time
        self.prev_frame_gray = frame_gray
        return frame_color

    def detect_and_describe(self, cv_image):
        # Get keypoints and descriptors.
        descriptor_data = self.feature_extraction.detect_and_describe(cv_image)
        xy = descriptor_data[:, :2]
        scores = descriptor_data[:, 2]
        scales = descriptor_data[:, 3]
        descriptors = descriptor_data[:, 4:]
        num_keypoints = descriptor_data.shape[0]

        if len(self.prev_xy) > 0:
            # Find new descriptors for the keypoints
            kdt = spatial.cKDTree(xy)
            dists, idxs = kdt.query(self.prev_xy)

            keep = dists < self.descriptor_reassociation_thr
            for i in range(keep.size):
                if keep[i]:
                    self.prev_descriptors[i] = descriptors[idxs[i]]

            self.prev_xy = self.prev_xy[keep]
            self.prev_scores = self.prev_scores[keep]
            self.prev_scales = self.prev_scales[keep]
            self.prev_descriptors = self.prev_descriptors[keep]
            self.prev_track_ids = self.prev_track_ids[keep]

            # Limit number of new detections added to fit with the global limit,
            # and mask detections to not initialize keypoints that are too close
            # to previous ones or to new ones
            quota = self.lk_num_keypoints - self.prev_xy.shape[0]
            mask = np.ones((cv_image.shape[0], cv_image.shape[1]))
            for prev_xy in self.prev_xy:
                x, y = prev_xy.astype(np.int32)
                cv2.circle(mask, (x, y), self.lk_mask_redetections_thr_px, 0,
                    cv2.FILLED)

            keep = []
            for i in range(xy.shape[0]):
                if quota <= 0:
                    break
                x, y = xy[i].astype(np.int32)
                if mask[y, x]:
                    keep.append(i)
                    quota -= 1
            keep = np.array(keep)
            if keep.size == 0:
                return

            # Assign new track ids
            track_ids = np.arange(
                self.next_track_id,
                self.next_track_id + keep.size).astype(np.int32)
            self.next_track_id += keep.size

            self.prev_xy = np.concatenate([self.prev_xy, xy[keep]])
            self.prev_scores = np.concatenate([self.prev_scores, scores[keep]])
            self.prev_scales = np.concatenate([self.prev_scales, scales[keep]])
            self.prev_descriptors = np.concatenate(
                [self.prev_descriptors, descriptors[keep]])
            self.prev_track_ids = np.concatenate(
                [self.prev_track_ids, track_ids])
        else:
            # Assign new track ids
            track_ids = np.arange(
                self.next_track_id,
                self.next_track_id + self.lk_num_keypoints).astype(np.int32)
            self.next_track_id += self.lk_num_keypoints

            self.prev_xy = xy[:self.lk_num_keypoints]
            self.prev_scores = scores[:self.lk_num_keypoints]
            self.prev_scales = scales[:self.lk_num_keypoints]
            self.prev_descriptors = descriptors[:self.lk_num_keypoints]
            self.prev_track_ids = track_ids[:self.lk_num_keypoints]

        for kp in xy:
            cv2.circle(cv_image, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
        cv2.imshow("Detections", cv_image)
        cv2.waitKey(3)

    def publish_features(self, stamp):
        num_keypoints = int(self.prev_xy.shape[0])

        # Flatten descriptors and convert to bytes
        descriptors = self.prev_descriptors.flatten().view(np.uint8)

        # Fill in basic message data
        feature_msg = Features()
        feature_msg.header.stamp = stamp
        feature_msg.numKeypointMeasurements = num_keypoints
        feature_msg.keypointMeasurementsX = self.prev_xy[:, 0].tolist()
        feature_msg.keypointMeasurementsY = self.prev_xy[:, 1].tolist()
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

        cv_image = self.lk_track(cv_image)
        self.detect_and_describe(cv_image)

        self.publish_features(image_msg.header.stamp)

if __name__ == '__main__':
    rospy.init_node('lk_tracker', anonymous=True)
    receiver = ImageReceiver()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down LK tracker.")
