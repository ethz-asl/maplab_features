#!/usr/bin/env python2
import cv2
import numpy as np
import threading

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from maplab_msgs.msg import Features

from config import LidarImageConfig
from point_cloud_utils import PointCloudUtils

class LidarReceiver:
    def __init__(self):
        self.config = LidarImageConfig()
        self.config.init_from_config()
        self.utils = PointCloudUtils(self.config)

        # Image processing
        self.intensity_clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 12))
        self.horizontal_filter_kernel = np.array([[0.125],
                                                  [0.25],
                                                  [0.25],
                                                  [0.25],
                                                  [0.125]])
        self.merge_mertens = cv2.createMergeMertens()
        self.cv_bridge = CvBridge()

        # Subscriber and publisher.
        self.pointcloud_sub = rospy.Subscriber(
            self.config.in_pointcloud_topic, PointCloud2,
            self.pointcloud_callback, queue_size=4000)
        self.feature_image_pub = rospy.Publisher(
            self.config.out_image_topic, Image, queue_size=4000)
        self.mask_image_pub = rospy.Publisher(
            self.config.out_mask_topic, Image, queue_size=4000)
        rospy.loginfo('[LidarReceiver] Subscribed to points {sub}.'.format(
            sub=self.config.in_pointcloud_topic))
        rospy.loginfo('[LidarReceiver] Publishing images on {pub}.'.format(
            pub=self.config.out_image_topic))
        rospy.loginfo('[LidarReceiver] Publishing image masks on {pub}.'.format(
            pub=self.config.out_image_topic))

        self.feature2D_sub = rospy.Subscriber(
            self.config.in_feature_topic, Features,
            self.feature_callback, queue_size=4000)
        self.feature3D_pub = rospy.Publisher(
            self.config.out_feature_topic, Features, queue_size=4000)
        rospy.loginfo('[LidarReceiver] Subscribed to 2D features {sub}.'.format(
            sub=self.config.in_pointcloud_topic))
        rospy.loginfo('[LidarReceiver] Publishing 3D features on {pub}.'.format(
            pub=self.config.out_image_topic))

        # Store LiDAR data between publishing the image and feature message
        self.proj_buffer = {}
        self.proj_buffer_mutex = threading.Lock()

    def pointcloud_callback(self, msg):
        cloud, time_offsets = self.utils.convert_msg_to_array(msg)

        range_img, intensity_img, inpaint_mask, proj_cloud, proj_time_offset = \
            self.utils.project_cloud_to_2d(cloud, time_offsets,
                self.config.projection_height, self.config.projection_width)
        feature_img = self.process_images(range_img, intensity_img, inpaint_mask)

        if self.config.visualize:
            self.visualize_projection(range_img, intensity_img)
            self.visualize_feature_image(feature_img)

            #inpaint_img = cv2.cvtColor(inpaint_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            #cv2.imshow("mask", inpaint_img)
            #cv2.waitKey(1)

        img_msg = self.cv_bridge.cv2_to_imgmsg(feature_img, "mono8")
        img_msg.header.stamp = msg.header.stamp
        self.feature_image_pub.publish(img_msg)

        mask_msg = self.cv_bridge.cv2_to_imgmsg(inpaint_mask, "mono8")
        mask_msg.header.stamp = msg.header.stamp
        self.mask_image_pub.publish(mask_msg)

        # Save 3D cloud information so we can later fill in the missing fields
        # in the corresponding Feature message that will be published later
        self.proj_buffer_mutex.acquire()
        self.proj_buffer[msg.header.stamp] = (proj_cloud, proj_time_offset)
        self.proj_buffer_mutex.release()

    def process_images(self, range_img, intensity_img, inpaint_mask):
        intensity_img = cv2.inpaint(
            intensity_img, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
        #cv2.imshow("intensity_inpaint", intensity_img)
        #cv2.waitKey(1)

        # Filter horizontal lines.
        intensity_img = cv2.filter2D(
            intensity_img, -1, self.horizontal_filter_kernel)
        intensity_img = cv2.medianBlur(intensity_img, 3)
        #cv2.imshow("intensity_filter", intensity_img)
        #cv2.waitKey(1)

        range_img = cv2.inpaint(range_img, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
        range_img = cv2.GaussianBlur(range_img, (3,3) ,cv2.BORDER_DEFAULT)

        range_grad_x = cv2.convertScaleAbs(
            cv2.Sobel(range_img, cv2.CV_8U, dx=1, dy=0, ksize=3))
        range_grad_y = cv2.convertScaleAbs(
            cv2.Sobel(range_img, cv2.CV_8U, dx=0, dy=1, ksize=3))
        range_gradient = cv2.addWeighted(range_grad_x, 0.5, range_grad_y, 0.5, 0)

        hdr_image = np.clip(
            self.merge_mertens.process([range_gradient, intensity_img]) * 255, 0, 255)

        return hdr_image.astype(np.uint8)

    def visualize_projection(self, range_img, intensity_img):
        range_img = cv2.cvtColor(range_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        intensity_img = cv2.cvtColor(intensity_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imshow("range", range_img)
        cv2.imshow("intensity", intensity_img)
        cv2.waitKey(1)

    def visualize_feature_image(self, feature_img):
        feature_img = cv2.cvtColor(feature_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imshow("feature", feature_img)
        cv2.waitKey(1)

    def feature_callback(self, msg):
        self.proj_buffer_mutex.acquire()
        proj_cloud, proj_time_offset = self.proj_buffer[msg.header.stamp]
        del self.proj_buffer[msg.header.stamp]
        self.proj_buffer_mutex.release()

        keypoint3DX = []
        keypoint3DY = []
        keypoint3DZ = []
        keypointTimeOffset = []
        for i in range(msg.numKeypointMeasurements):
            img_x = int(round(msg.keypointMeasurementsX[i]))
            img_y = int(round(msg.keypointMeasurementsY[i]))

            xyz = proj_cloud[img_y, img_x]
            keypoint3DX.append(xyz[0])
            keypoint3DY.append(xyz[1])
            keypoint3DZ.append(xyz[2])

            time_offset = proj_time_offset[img_y, img_x]
            keypointTimeOffset.append(time_offset)

        msg.keypoint3DX = keypoint3DX
        msg.keypoint3DY = keypoint3DY
        msg.keypoint3DZ = keypoint3DZ
        msg.keypointTimeOffset = keypointTimeOffset

        self.feature3D_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('lidar_image_converter', anonymous=True)
    receiver = LidarReceiver()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down LiDAR image converter.")
