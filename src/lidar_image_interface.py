#!/usr/bin/env python2
import numpy as np

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from maplab_msgs.msg import Features
from point_cloud_utils import *
import matplotlib.pyplot as plt
import message_filters

from config import LidarImageConfig
from point_cloud_utils import PointCloudUtils

class LidarReceiver:
    def __init__(self):
        self.config = LidarImageConfig()
        self.config.init_from_config()
        self.utils = PointCloudUtils(self.config)

        # Image processing
        self.intensity_clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 12))
        self.horizontal_filter_kernel = np.array([[0, 0, 0.125],
                                                  [0, 0, 0.25],
                                                  [0, 0, 0.25],
                                                  [0, 0, 0.25],
                                                  [0, 0, 0.125]])
        self.merge_mertens = cv2.createMergeMertens()

        # Subscriber and publisher.
        if self.config.operation_mode == 'projection':
            self.pc_sub = rospy.Subscriber(self.config.in_pointcloud_topic, PointCloud2, self.pointcloud_callback)
            rospy.loginfo('[LidarReceiver] Subscribed to {sub}.'.format(sub=self.config.in_pointcloud_topic))
        elif self.config.operation_mode == 'image':
            range_img_sub = message_filters.Subscriber(self.config.in_range_image_topic, Image)
            intensity_img_sub = message_filters.Subscriber(self.config.in_intensity_image_topic, Image)
            self.ts = message_filters.TimeSynchronizer([range_img_sub, intensity_img_sub], 10)
            self.ts.registerCallback(self.image_callback)
            rospy.loginfo('[LidarReceiver] Subscribed to {range_sub} and {intensity_sub}.'.format(range_sub=self.config.in_range_image_topic, intensity_sub=self.config.in_range_image_topic))

        self.feature_image_pub = rospy.Publisher(
                self.config.out_image_topic, Image, queue_size=20)
        self.bridge = CvBridge()
        rospy.loginfo('[LidarReceiver] Publishing on {pub}.'.format(pub=self.config.out_image_topic))

    def pointcloud_callback(self, msg):
        cloud = self.utils.convert_msg_to_array(msg)

        range_img, intensity_img = self.utils.project_cloud_to_2d(cloud, self.config.fov_up, self.config.fov_down, self.config.projection_height, self.config.projection_width)
        self.merge_and_publish_feature_image(range_img, intensity_img)

    def image_callback(self, range_msg, intensity_msg):
        range_img = self.bridge.imgmsg_to_cv2(range_msg)
        intensity_img = self.bridge.imgmsg_to_cv2(intensity_msg)
        self.merge_and_publish_feature_image(range_img, intensity_img)

    def merge_and_publish_feature_image(self, range_img, intensity_img):
        feature_img = self.process_images(range_img, intensity_img)
        if self.config.visualize:
            self.visualize_projection(range_img, intensity_img)
            self.visualize_feature_image(feature_img)

        img_msg = self.bridge.cv2_to_imgmsg(feature_img, "mono8")
        self.feature_image_pub.publish(img_msg)

    def process_images(self, range_img, intensity_img):
        # Perform a histogram equalization of the intensity channel
        intensity_img = self.intensity_clahe.apply(intensity_img)

        # Filter horizontal lines.
        intensity_img = cv2.filter2D(intensity_img, -1, self.horizontal_filter_kernel)

        inpaint_mask = self.utils.compute_inpaint_mask(range_img)
        range_img = cv2.inpaint(range_img, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
        range_img = cv2.GaussianBlur(range_img, (3,3) ,cv2.BORDER_DEFAULT)

        range_grad_x = cv2.convertScaleAbs(cv2.Sobel(range_img, cv2.CV_8U, dx=1, dy=0, ksize=3))
        range_grad_y = cv2.convertScaleAbs(cv2.Sobel(range_img, cv2.CV_8U, dx=0, dy=1, ksize=3))

        range_gradient = cv2.addWeighted(range_grad_x, 0.5, range_grad_y, 0.5, 0)

        hdr_image = np.clip(self.merge_mertens.process([range_gradient, intensity_img]) * 255, 0, 255)

        if self.config.resize_output:
            hdr_image = cv2.resize(hdr_image, (4096, 256), 0, 0, interpolation=cv2.INTER_NEAREST)

        return hdr_image.astype(np.uint8)


    def visualize_projection(self, range_img, intensity_img):
        range_img = cv2.cvtColor(range_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        intensity_img = cv2.cvtColor(intensity_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # range_img_c = cv2.applyColorMap(range_img, cv2.COLORMAP_JET)
        cv2.imshow("range", range_img)
        cv2.imshow("intensity", intensity_img)
        cv2.waitKey(1)

    def visualize_feature_image(self, feature_img):
        feature_img = cv2.cvtColor(feature_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imshow("feature", feature_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('lidar_image_converter', anonymous=True)
    receiver = LidarReceiver()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down LiDAR image converter.")
