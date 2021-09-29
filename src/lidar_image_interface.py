#!/usr/bin/env python2
import numpy as np

import cv2
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from maplab_msgs.msg import Features
from point_cloud_utils import *
import matplotlib.pyplot as plt

from config import LidarImageConfig
from point_cloud_utils import PointCloudUtils

class LidarReceiver:
    def __init__(self, lidar_topic, image_topic):
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
        self.pc_sub = rospy.Subscriber(lidar_topic, PointCloud2, self.pointcloud_callback)
        self.descriptor_pub = rospy.Publisher(
                image_topic, Image, queue_size=20)
        rospy.loginfo('[LidarReceiver] Subscribed to {sub}.'.format(sub=lidar_topic))
        rospy.loginfo('[LidarReceiver] Publishing on {pub}.'.format(pub=image_topic))

    def pointcloud_callback(self, msg):
        cloud = self.utils.convert_msg_to_array(msg)

        range_img, intensity_img, inpaint_mask = self.utils.project_cloud_to_2d(cloud, self.config.fov_up, self.config.fov_down, self.config.projection_height, self.config.projection_width)
        feature_img = self.process_images(range_img, intensity_img, inpaint_mask)
        if self.config.visualize:
            self.visualize_projection(range_img, intensity_img)
            self.visualize_feature_image(feature_img)


    def process_images(self, range_img, intensity_img, inpaint_mask):
        # Perform a histogram equalization of the intensity channel
        intensity_img = self.intensity_clahe.apply(intensity_img)

        # Filter horizontal lines.
        intensity_img = cv2.filter2D(intensity_img, -1, self.horizontal_filter_kernel)

        range_img = cv2.inpaint(range_img, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
        range_img = cv2.GaussianBlur(range_img, (3,3) ,cv2.BORDER_DEFAULT)


        range_grad_x = cv2.convertScaleAbs(cv2.Sobel(range_img, cv2.CV_8U, dx=1, dy=0, ksize=3))
        range_grad_y = cv2.convertScaleAbs(cv2.Sobel(range_img, cv2.CV_8U, dx=0, dy=1, ksize=3))

        range_gradient = cv2.addWeighted(range_grad_x, 0.5, range_grad_y, 0.5, 0)

        hdr_image = self.merge_mertens.process([range_gradient, intensity_img]) * 255
        return hdr_image


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
    receiver = LidarReceiver('/os_cloud_node/points', '/os_cloud_node/images')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
