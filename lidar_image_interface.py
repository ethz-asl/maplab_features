#!/usr/bin/env python2
import numpy as np

import cv2
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from maplab_msgs.msg import Features
from point_cloud_utils import *
import matplotlib.pyplot as plt

class LidarReceiver:
    def __init__(self, lidar_topic, image_topic):
        # Subscriber and publisher.
        self.pc_sub = rospy.Subscriber(lidar_topic, PointCloud2, self.pointcloud_callback)
        self.descriptor_pub = rospy.Publisher(
                image_topic, Image, queue_size=20)
        rospy.loginfo('[LidarReceiver] Subscribed to {sub}.'.format(sub=lidar_topic))
        rospy.loginfo('[LidarReceiver] Publishing on {pub}.'.format(pub=image_topic))

    def pointcloud_callback(self, msg):
        rospy.loginfo('[LidarReceiver] Received pointcloud.')
        cloud = convert_msg_to_array(msg)

        # TODO(lbern): introduce configs
        # OS-1
        #fov_up=17.5
        #fov_down=-15.5
        proj_H = 64
        proj_W = 1024

        # OS-0
        fov_up=50.5
        fov_down=-47.5
        proj_H = 128
        proj_W = 1024

        proj_range, _, proj_intensity, _ = project_cloud_to_2d(cloud, fov_up, fov_down, proj_H, proj_W)
        self.visualize_projection(proj_range, proj_intensity)


    def visualize_projection(self, range_img, intensity_img):
        fig, axs = plt.subplots(2, figsize=(12, 8))
        plt.ion()
        plt.show(block=False)

        axs[0].set_title('range')
        axs[0].imshow(range_img)
        axs[0].set_axis_off()

        axs[1].set_title('intensity')
        intensity_img[intensity_img < 0] = 0
        axs[1].imshow(intensity_img, cmap='gray')
        axs[1].set_axis_off()


if __name__ == '__main__':
    rospy.init_node('lidar_image_converter', anonymous=True)
    receiver = LidarReceiver('/os_cloud_node/points', '/os_cloud_node/images')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
