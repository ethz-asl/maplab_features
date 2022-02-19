#!/usr/bin/env python
from __future__ import print_function

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
from utils_py2 import open_fifo, read_np, send_np
from feature_extraction import FeatureExtractionCv, FeatureExtractionExternal

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

        self.tracker = 'lk'

        if self.tracker == 'lk':
            # LK tracker settings
            self.lk_params = dict(
                winSize  = (15, 15),
                maxLevel = 2,
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
            self.lk_max_step_px = 2.0
            self.lk_merge_tracks_thr_px = 3
        elif self.tracker == 'superglue':
            # Pipe for transferring images
            self.fifo_images = open_fifo('/tmp/maplab_superglue_images', 'wb')
            self.fifo_matches = open_fifo('/tmp/maplab_superglue_matches', 'rb')
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

    def lk_track(self, frame_gray):
        # Check if there is anything to track
        if len(self.prev_xy) > 0 and len(self.prev_frame) > 0:
            # Use optical from to determine keypoint movement
            p0 = self.prev_xy.reshape(-1, 1, 2).astype(np.float32)
            p1, _, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, frame_gray, p0, None, **self.lk_params)
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(
                frame_gray, self.prev_frame, p1, None, **self.lk_params)

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
            keep = np.array(keep).astype(np.int)

            if self.debug:
                # Visualization
                old_xy = self.prev_xy[keep].astype(np.int)
                new_xy = p1[keep].reshape(-1, 2).astype(np.int)
                vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                for xy0, xy1 in zip(old_xy, new_xy):
                    cv2.line(vis, tuple(xy0), tuple(xy1), (0, 255, 0), 1)
                for kp in new_xy:
                    cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)

                cv2.imshow("Tracking", vis)
                cv2.waitKey(3)

            # Drop bad keypoints
            if keep.size != 0:
                self.prev_xy = p1[keep].reshape(-1, 2)
                self.prev_scales = self.prev_scales[keep]
                self.prev_scores = self.prev_scores[keep]
                self.prev_descriptors = self.prev_descriptors[keep]
                self.prev_track_ids = self.prev_track_ids[keep]
            else:
                self.prev_xy = []
                self.prev_scales = []
                self.prev_scores = []
                self.prev_descriptors = []
                self.prev_track_ids = []

        # Save current frame for next time
        self.prev_frame = frame_gray

    def superglue_track(self, frame_gray):
        # Check if there is anything to track
        if len(self.prev_xy) > 0 and len(self.prev_frame) > 0:
            # Transmit images for processing
            cv_success0, cv_binary0 = cv2.imencode('.png', self.prev_frame)
            cv_success1, cv_binary1 = cv2.imencode('.png', frame_gray)
            assert(cv_success0 and cv_success1)
            send_np(self.fifo_images, cv_binary0)
            send_np(self.fifo_images, cv_binary1)

            send_np(self.fifo_images, self.prev_xy)
            send_np(self.fifo_images, self.prev_scores)
            send_np(self.fifo_images, self.prev_descriptors)

            send_np(self.fifo_images, self.xy)
            send_np(self.fifo_images, self.scores)
            send_np(self.fifo_images, self.descriptors)

            matches = read_np(self.fifo_matches, np.int32)
            valid = matches > -1

            if self.debug:
                # Visualization
                mxy0 = self.prev_xy[valid]
                mxy1 = self.xy[matches[valid]]

                vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                for xy0, xy1 in zip(mxy0, mxy1):
                    cv2.line(vis, tuple(xy0), tuple(xy1), (0, 255, 0), 1)
                for kp in mxy1:
                    cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)

                cv2.imshow("Tracking", vis)
                cv2.waitKey(3)

            self.prev_xy = self.xy[matches[valid]]
            self.prev_scores = self.scores[matches[valid]]
            self.prev_scales = self.scales[matches[valid]]
            self.prev_descriptors = self.descriptors[matches[valid]]
            self.prev_track_ids = self.prev_track_ids[valid]

        self.prev_frame = frame_gray

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

        self.detect_and_describe(frame_color)

        if self.tracker == 'lk':
            self.lk_track(frame_gray)
        elif self.tracker == 'superglue':
            self.superglue_track(frame_gray)

        if len(self.prev_xy) > 0:
            # Limit number of new detections added to fit with the global limit,
            # and mask detections to not initialize keypoints that are too close
            # to previous ones or to new ones
            quota = self.num_keypoints - self.prev_xy.shape[0]
            mask = np.ones((cv_image.shape[0], cv_image.shape[1]))
            for prev_xy in self.prev_xy:
                x, y = prev_xy.astype(np.int32)
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
            if keep.size == 0:
                return

            # Assign new track ids
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


        print('r', self.count_recv)

        if len(self.prev_xy) > 0:
            self.publish_features(image_msg.header.stamp)

if __name__ == '__main__':
    rospy.init_node('lk_tracker', anonymous=True)
    receiver = ImageReceiver()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")
